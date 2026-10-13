import json
import tempfile
import unittest
from pathlib import Path

from rag_project.domain.manifest import IndexManifest, SourceFileRecord
from rag_project.domain.models import Document
from rag_project.infrastructure.faiss_store import FaissVectorStore
from rag_project.infrastructure.index_storage import (
    INDEX_FILES,
    FaissIndexRepository,
    IndexStats,
)


def build_index(index_dir: Path, count: int = 2) -> FaissVectorStore:
    store = FaissVectorStore(index_dir)
    store.add(
        [
            Document(id=f"doc-{i}:0000", text=f"text {i}", metadata={"relative_path": f"doc-{i}"})
            for i in range(count)
        ],
        [[1.0, float(i)] for i in range(count)],
    )
    store.save()
    return store


def sample_manifest() -> IndexManifest:
    return IndexManifest(
        files={
            "docs/manual.pdf": SourceFileRecord(
                path="docs/manual.pdf",
                sha256="a" * 64,
                size=2048,
                mtime_ns=1_700_000_000_000_000_000,
                chunk_ids=["docs/manual.pdf:page-0001:0000", "docs/manual.pdf:page-0002:0000"],
            ),
            "заметки.md": SourceFileRecord(
                path="заметки.md", sha256="b" * 64, size=12, mtime_ns=5, chunk_ids=[]
            ),
        }
    )


class ExistsAndStoreTests(unittest.TestCase):
    def test_missing_directory_does_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(FaissIndexRepository(Path(temp_dir) / "missing").exists())

    def test_one_index_file_alone_is_not_an_index(self) -> None:
        for present in ("index.faiss", "documents.json"):
            with self.subTest(present=present), tempfile.TemporaryDirectory() as temp_dir:
                (Path(temp_dir) / present).write_text("x", encoding="utf-8")
                self.assertFalse(FaissIndexRepository(temp_dir).exists())

    def test_saved_index_exists_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index(Path(temp_dir), count=3)
            repository = FaissIndexRepository(temp_dir)

            self.assertTrue(repository.exists())
            self.assertEqual(repository.load_store().count(), 3)

    def test_new_store_is_empty_and_targets_the_index_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FaissIndexRepository(temp_dir).new_store()

            self.assertEqual(store.count(), 0)
            self.assertEqual(store.index_dir, Path(temp_dir))

    def test_load_store_without_index_raises_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                FaissIndexRepository(temp_dir).load_store()


class ManifestStorageTests(unittest.TestCase):
    def test_manifest_round_trip_preserves_every_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = FaissIndexRepository(Path(temp_dir) / "not-yet-created")
            manifest = sample_manifest()

            repository.save_manifest(manifest)

            self.assertEqual(repository.load_manifest(), manifest)

    def test_manifest_version_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = FaissIndexRepository(temp_dir)

            repository.save_manifest(IndexManifest(files={}, version=3))

            self.assertEqual(repository.load_manifest().version, 3)

    def test_missing_manifest_loads_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = FaissIndexRepository(temp_dir).load_manifest()

            self.assertEqual(manifest, IndexManifest.empty())
            self.assertEqual(manifest.files, {})

    def test_corrupt_manifest_loads_as_empty_so_next_ingest_rebuilds(self) -> None:
        for content in ("{not json", '["a", "list"]', '{"files": {"a.txt": {"path": 1}}}'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temp_dir:
                (Path(temp_dir) / "manifest.json").write_text(content, encoding="utf-8")

                with self.assertLogs("rag_project.infrastructure.index_storage", "WARNING"):
                    manifest = FaissIndexRepository(temp_dir).load_manifest()

                self.assertEqual(manifest.files, {})


class ClearTests(unittest.TestCase):
    def test_clear_removes_index_files_and_keeps_others(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir)
            build_index(index_dir)
            repository = FaissIndexRepository(index_dir)
            repository.save_manifest(sample_manifest())
            (index_dir / "keep.txt").write_text("not an index file", encoding="utf-8")

            removed = repository.clear()

            self.assertEqual(removed, 3)
            for name in INDEX_FILES:
                self.assertFalse((index_dir / name).exists(), name)
            self.assertTrue((index_dir / "keep.txt").exists())
            self.assertFalse(repository.exists())

    def test_clear_counts_only_files_that_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index(Path(temp_dir))  # no manifest written
            repository = FaissIndexRepository(temp_dir)

            self.assertEqual(repository.clear(), 2)
            self.assertEqual(repository.clear(), 0)

    def test_clear_missing_directory_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(FaissIndexRepository(Path(temp_dir) / "missing").clear(), 0)


class StatsTests(unittest.TestCase):
    def test_chunk_count_for_current_documents_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index(Path(temp_dir), count=3)
            payload = json.loads((Path(temp_dir) / "documents.json").read_text(encoding="utf-8"))
            self.assertIsInstance(payload, dict)  # sanity: this is the current format

            self.assertEqual(FaissIndexRepository(temp_dir).stats().chunk_count, 3)

    def test_chunk_count_for_legacy_bare_list_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy = [
                {"id": "a:0000", "text": "alpha", "metadata": {}},
                {"id": "b:0000", "text": "beta", "metadata": {}},
            ]
            (Path(temp_dir) / "documents.json").write_text(json.dumps(legacy), encoding="utf-8")

            self.assertEqual(FaissIndexRepository(temp_dir).stats().chunk_count, 2)

    def test_missing_documents_file_has_unknown_count_and_zero_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                FaissIndexRepository(temp_dir).stats(), IndexStats(chunk_count=None, size_bytes=0)
            )
            self.assertEqual(
                FaissIndexRepository(Path(temp_dir) / "missing").stats(),
                IndexStats(chunk_count=None, size_bytes=0),
            )

    def test_corrupt_documents_file_has_unknown_count(self) -> None:
        for raw in ("{not json", '{"faiss_ids": [0, 1]}', "42"):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temp_dir:
                (Path(temp_dir) / "documents.json").write_text(raw, encoding="utf-8")

                stats = FaissIndexRepository(temp_dir).stats()

                self.assertIsNone(stats.chunk_count)
                self.assertEqual(stats.size_bytes, len(raw.encode("utf-8")))

    def test_size_bytes_sums_only_index_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir)
            build_index(index_dir)
            repository = FaissIndexRepository(index_dir)
            repository.save_manifest(sample_manifest())
            (index_dir / "unrelated.bin").write_bytes(b"x" * 10_000)

            expected = sum((index_dir / name).stat().st_size for name in INDEX_FILES)

            self.assertGreater(expected, 0)
            self.assertEqual(repository.stats().size_bytes, expected)
