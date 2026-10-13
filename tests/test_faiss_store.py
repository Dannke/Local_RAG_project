import tempfile
import unittest
from pathlib import Path

from rag_project.application.indexing import remove_document
from rag_project.domain.manifest import IndexManifest, SourceFileRecord
from rag_project.domain.models import Document
from rag_project.infrastructure.faiss_store import FaissVectorStore
from rag_project.infrastructure.index_storage import FaissIndexRepository


class FaissVectorStoreTests(unittest.TestCase):
    def test_save_load_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FaissVectorStore(temp_dir)
            store.add(
                [
                    Document(id="a", text="alpha document"),
                    Document(id="b", text="beta document"),
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            )
            store.save()

            loaded = FaissVectorStore(temp_dir)
            loaded.load()
            results = loaded.search([1.0, 0.0], top_k=1)

            self.assertEqual(loaded.count(), 2)
            self.assertEqual(results[0].document.id, "a")

    def test_load_from_disk_classmethod(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FaissVectorStore(temp_dir)
            store.add([Document(id="a", text="alpha document")], [[1.0, 0.0]])
            store.save()

            loaded = FaissVectorStore.load_from_disk(temp_dir)

            self.assertEqual(loaded.count(), 1)
            self.assertEqual(loaded.search([1.0, 0.0], top_k=1)[0].document.text, "alpha document")

    def test_remove_one_document_keeps_others_intact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FaissVectorStore(temp_dir)
            store.add(
                [
                    Document(id="alpha", text="first document"),
                    Document(id="beta", text="second document"),
                    Document(id="gamma", text="third document"),
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            )

            removed = store.remove_ids(["beta"])

            self.assertEqual(removed, 1)
            self.assertEqual(store.count(), 2)
            self.assertEqual(sorted(doc.id for doc in store.documents), ["alpha", "gamma"])

            # Removing the middle vector must not disturb the survivors' search.
            results = store.search([1.0, 0.0, 0.0], top_k=2)
            self.assertEqual([r.document.id for r in results], ["alpha", "gamma"])

    def test_remove_survives_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FaissVectorStore(temp_dir)
            store.add(
                [
                    Document(id="a", text="dog document"),
                    Document(id="b", text="cat document"),
                    Document(id="c", text="bird document"),
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, -1.0],
                ],
            )
            store.remove_ids(["b"])
            store.save()

            loaded = FaissVectorStore.load_from_disk(temp_dir)
            self.assertEqual(loaded.count(), 2)
            self.assertEqual(sorted(doc.id for doc in loaded.documents), ["a", "c"])
            results = loaded.search([1.0, 0.0], top_k=2)
            self.assertEqual([r.document.id for r in results], ["a", "c"])

    def test_remove_non_existent_id_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FaissVectorStore(temp_dir)
            store.add([Document(id="a", text="alpha")], [[1.0, 0.0]])
            self.assertEqual(store.remove_ids(["missing"]), 0)
            self.assertEqual(store.count(), 1)

    def test_remove_then_add_reuses_no_stale_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FaissVectorStore(temp_dir)
            store.add([Document(id="a", text="alpha")], [[1.0, 0.0]])
            store.add([Document(id="b", text="beta")], [[0.0, 1.0]])
            store.remove_ids(["a"])
            store.add([Document(id="c", text="gamma")], [[-1.0, 0.0]])

            self.assertEqual(store.count(), 2)
            results = store.search([0.0, 1.0], top_k=1)
            self.assertEqual(results[0].document.id, "b")


class DeleteDocumentFromIndexTests(unittest.TestCase):
    def _seed_index(self, index_dir) -> FaissVectorStore:
        store = FaissVectorStore(index_dir)
        store.add(
            [
                Document(
                    id="docs/manual.pdf",
                    text="manual content",
                    metadata={"relative_path": "docs/manual.pdf"},
                ),
                Document(
                    id="docs/readme.md",
                    text="readme content",
                    metadata={"relative_path": "docs/readme.md"},
                ),
            ],
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        store.save()
        manifest = IndexManifest(
            files={
                "docs/manual.pdf": SourceFileRecord(
                    path="docs/manual.pdf", sha256="a", size=1, mtime_ns=1,
                    chunk_ids=["docs/manual.pdf"],
                ),
                "docs/readme.md": SourceFileRecord(
                    path="docs/readme.md", sha256="b", size=1, mtime_ns=1,
                    chunk_ids=["docs/readme.md"],
                ),
            }
        )
        FaissIndexRepository(index_dir).save_manifest(manifest)
        return store

    def test_delete_removes_only_target_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self._seed_index(temp_dir)

            removed = remove_document(FaissIndexRepository(temp_dir), "docs/readme.md")

            self.assertEqual(removed, 1)
            store = FaissVectorStore.load_from_disk(temp_dir)
            self.assertEqual(store.count(), 1)
            self.assertEqual(store.documents[0].id, "docs/manual.pdf")

            manifest = FaissIndexRepository(temp_dir).load_manifest()
            self.assertNotIn("docs/readme.md", manifest.files)
            self.assertIn("docs/manual.pdf", manifest.files)

    def test_delete_unknown_file_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self._seed_index(temp_dir)
            self.assertEqual(remove_document(FaissIndexRepository(temp_dir), "missing.txt"), 0)

    def test_delete_without_index_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(remove_document(FaissIndexRepository(temp_dir), "x.pdf"), 0)

    def test_delete_with_unreadable_manifest_still_removes_vectors(self) -> None:
        # Regression: an unreadable manifest has no record for the file, so the
        # chunks must be found by metadata instead of silently staying indexed.
        with tempfile.TemporaryDirectory() as temp_dir:
            self._seed_index(temp_dir)
            (Path(temp_dir) / "manifest.json").write_text("{broken", encoding="utf-8")

            with self.assertLogs(level="WARNING"):
                removed = remove_document(FaissIndexRepository(temp_dir), "docs/readme.md")

            self.assertEqual(removed, 1)
            store = FaissVectorStore.load_from_disk(temp_dir)
            self.assertEqual([doc.id for doc in store.documents], ["docs/manual.pdf"])
