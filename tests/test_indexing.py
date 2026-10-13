from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from rag_project.application.indexing import DocumentIndexer, remove_document
from rag_project.domain.errors import NoDocumentsToIndexError
from rag_project.domain.manifest import IndexManifest, SourceFileRecord
from rag_project.domain.models import Document, SearchResult
from rag_project.infrastructure.document_directory import DocumentDirectory
from rag_project.infrastructure.embeddings import HashingEmbeddingModel
from rag_project.infrastructure.index_storage import FaissIndexRepository

CHUNK_SIZE = 60
CHUNK_OVERLAP = 0

# Three ~54-character paragraphs: with a 60-character budget each becomes its own chunk.
THREE_PARAGRAPHS = (
    "Alpha paragraph talks about vector indexes and FAISS.\n\n"
    "Beta paragraph explains how chunking splits documents.\n\n"
    "Gamma paragraph covers the incremental manifest logic."
)
A_CHUNK_IDS = ["a.txt:0000", "a.txt:0001", "a.txt:0002"]
SHORT_NOTE = "Short note about cats."

FULL_BUILD_STAGES = [
    "Загрузка документов",
    "Разделение на фрагменты",
    "Инициализация индекса",
    "Индексирование фрагментов",
    "Сохранение индекса",
]


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CountingEmbedder:
    """Deterministic offline embedder that records every ``embed_texts`` call."""

    def __init__(self) -> None:
        self._inner = HashingEmbeddingModel(dimensions=16)
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return self._inner.embed_texts(texts)

    @property
    def embedded_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


class FakeDocumentSource:
    """In-memory ``DocumentSource``: maps a relative path to the file's text."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = dict(files)
        self.load_all_calls = 0
        self.loaded_paths: list[str] = []

    def scan(self) -> dict[str, SourceFileRecord]:
        return {
            path: SourceFileRecord(
                path=path,
                sha256=sha256_of(text),
                size=len(text.encode("utf-8")),
                mtime_ns=0,
            )
            for path, text in sorted(self.files.items())
        }

    def load_all(self) -> list[Document]:
        self.load_all_calls += 1
        return [document for path in sorted(self.files) for document in self._documents(path)]

    def load(self, relative_path: str) -> list[Document]:
        self.loaded_paths.append(relative_path)
        return self._documents(relative_path)

    def _documents(self, path: str) -> list[Document]:
        text = self.files[path]
        if not text.strip():
            return []
        return [Document(id=path, text=text, metadata={"relative_path": path})]


class FakeStore:
    """``PersistentVectorStore`` whose ``save()`` persists into its repository."""

    def __init__(self, repository: FakeIndexRepository, documents: list[Document]) -> None:
        self._repository = repository
        self.documents = list(documents)

    def add(self, documents: Sequence[Document], embeddings: Sequence[Sequence[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")
        self.documents.extend(documents)

    def search(self, embedding: Sequence[float], top_k: int = 5) -> list[SearchResult]:
        return []

    def count(self) -> int:
        return len(self.documents)

    def remove_ids(self, document_ids: Sequence[str]) -> int:
        remove = set(document_ids)
        before = len(self.documents)
        self.documents = [document for document in self.documents if document.id not in remove]
        return before - len(self.documents)

    def ids_for_source(self, relative_path: str) -> list[str]:
        return [d.id for d in self.documents if d.relative_path == relative_path]

    def save(self) -> None:
        self._repository.persisted = list(self.documents)
        self._repository.save_calls += 1


class FakeIndexRepository:
    """In-memory ``IndexRepository``; ``persisted is None`` means no index on disk."""

    def __init__(self) -> None:
        self.persisted: list[Document] | None = None
        self.manifest = IndexManifest.empty()
        self.new_store_calls = 0
        self.load_store_calls = 0
        self.save_calls = 0

    def exists(self) -> bool:
        return self.persisted is not None

    def new_store(self) -> FakeStore:
        self.new_store_calls += 1
        return FakeStore(self, [])

    def load_store(self) -> FakeStore:
        if self.persisted is None:
            raise FileNotFoundError("no saved index")
        self.load_store_calls += 1
        return FakeStore(self, self.persisted)

    def load_manifest(self) -> IndexManifest:
        return self.manifest

    def save_manifest(self, manifest: IndexManifest) -> None:
        self.manifest = manifest

    @property
    def persisted_ids(self) -> list[str]:
        return [document.id for document in self.persisted or []]

    @property
    def persisted_texts(self) -> list[str]:
        return [document.text for document in self.persisted or []]


def make_indexer(
    source: FakeDocumentSource,
    repository: FakeIndexRepository,
    embedder: CountingEmbedder | None = None,
    token_counter=None,
) -> DocumentIndexer:
    return DocumentIndexer(
        embedding_model=embedder or CountingEmbedder(),
        sources=source,
        index=repository,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        token_counter=token_counter,
    )


def build_index(files: dict[str, str]) -> tuple[FakeDocumentSource, FakeIndexRepository]:
    source = FakeDocumentSource(files)
    repository = FakeIndexRepository()
    make_indexer(source, repository).ingest(incremental=False)
    return source, repository


class FullBuildTests(unittest.TestCase):
    def test_full_build_indexes_every_chunk_and_returns_count(self) -> None:
        source = FakeDocumentSource({"a.txt": THREE_PARAGRAPHS, "b.md": SHORT_NOTE})
        repository = FakeIndexRepository()
        embedder = CountingEmbedder()

        count = make_indexer(source, repository, embedder).ingest(incremental=False)

        self.assertEqual(count, 4)
        self.assertEqual(repository.persisted_ids, [*A_CHUNK_IDS, "b.md:0000"])
        self.assertEqual(len(embedder.embedded_texts), 4)
        self.assertEqual(repository.new_store_calls, 1)
        self.assertEqual(repository.save_calls, 1)
        self.assertEqual(source.load_all_calls, 1)

    def test_manifest_covers_every_scanned_file_including_files_without_chunks(self) -> None:
        source = FakeDocumentSource({"a.txt": THREE_PARAGRAPHS, "empty.txt": "   \n"})
        repository = FakeIndexRepository()

        make_indexer(source, repository).ingest(incremental=False)

        files = repository.manifest.files
        self.assertEqual(set(files), {"a.txt", "empty.txt"})
        self.assertEqual(files["a.txt"].chunk_ids, A_CHUNK_IDS)
        self.assertEqual(files["a.txt"].sha256, sha256_of(THREE_PARAGRAPHS))
        self.assertEqual(files["empty.txt"].chunk_ids, [])

    def test_full_build_reports_progress_stages_in_order(self) -> None:
        source = FakeDocumentSource({"a.txt": THREE_PARAGRAPHS, "b.md": SHORT_NOTE})
        events: list[tuple[str, int, int]] = []

        make_indexer(source, FakeIndexRepository()).ingest(
            incremental=False, progress=lambda *event: events.append(event)
        )

        self.assertEqual([stage for stage, _, _ in events], FULL_BUILD_STAGES)
        self.assertIn(("Индексирование фрагментов", 0, 4), events)

    def test_empty_source_raises_and_saves_nothing(self) -> None:
        for incremental in (True, False):
            with self.subTest(incremental=incremental):
                repository = FakeIndexRepository()
                embedder = CountingEmbedder()
                indexer = make_indexer(FakeDocumentSource({}), repository, embedder)

                with self.assertRaises(NoDocumentsToIndexError):
                    indexer.ingest(incremental=incremental)

                self.assertIsNone(repository.persisted)
                self.assertEqual(repository.manifest.files, {})
                self.assertEqual(embedder.calls, [])

    def test_source_with_only_blank_files_raises(self) -> None:
        indexer = make_indexer(FakeDocumentSource({"blank.txt": "  \n\t"}), FakeIndexRepository())

        with self.assertRaises(NoDocumentsToIndexError):
            indexer.ingest(incremental=False)

    def test_token_counter_is_used_for_chunk_size(self) -> None:
        # 24 words fit a 60-"token" budget, so the three paragraphs share one chunk.
        source = FakeDocumentSource({"a.txt": THREE_PARAGRAPHS})
        repository = FakeIndexRepository()

        count = make_indexer(
            source, repository, token_counter=lambda text: len(text.split())
        ).ingest(incremental=False)

        self.assertEqual(count, 1)
        self.assertEqual(repository.persisted_ids, ["a.txt:0000"])


class IncrementalIngestTests(unittest.TestCase):
    def test_nothing_changed_is_a_noop_that_never_embeds(self) -> None:
        source, repository = build_index({"a.txt": THREE_PARAGRAPHS, "b.md": SHORT_NOTE})
        manifest_before = repository.manifest
        embedder = CountingEmbedder()

        count = make_indexer(source, repository, embedder).ingest(incremental=True)

        self.assertEqual(count, 4)
        self.assertEqual(embedder.calls, [])
        self.assertEqual(source.load_all_calls, 1)  # only the initial build
        self.assertEqual(source.loaded_paths, [])
        self.assertEqual(repository.new_store_calls, 1)
        self.assertEqual(repository.save_calls, 1)
        self.assertEqual(repository.manifest, manifest_before)

    def test_new_file_is_appended_without_reembedding_existing_chunks(self) -> None:
        source, repository = build_index({"a.txt": THREE_PARAGRAPHS})
        a_record = repository.manifest.files["a.txt"]
        source.files["b.md"] = SHORT_NOTE
        embedder = CountingEmbedder()
        events: list[tuple[str, int, int]] = []

        count = make_indexer(source, repository, embedder).ingest(
            incremental=True, progress=lambda *event: events.append(event)
        )

        self.assertEqual(count, 4)
        self.assertEqual(embedder.embedded_texts, [SHORT_NOTE])
        self.assertEqual(source.loaded_paths, ["b.md"])
        self.assertEqual(source.load_all_calls, 1)
        self.assertEqual(repository.new_store_calls, 1)  # existing store was extended
        self.assertEqual(repository.persisted_ids, [*A_CHUNK_IDS, "b.md:0000"])

        files = repository.manifest.files
        self.assertEqual(set(files), {"a.txt", "b.md"})
        self.assertEqual(files["a.txt"], a_record)
        self.assertEqual(files["b.md"].chunk_ids, ["b.md:0000"])
        self.assertEqual(files["b.md"].sha256, sha256_of(SHORT_NOTE))

        self.assertIn(("Загрузка новых документов", 1, 1), events)
        self.assertEqual(events[-1][0], "Сохранение индекса")

    def test_added_file_without_text_does_not_touch_the_index(self) -> None:
        source, repository = build_index({"a.txt": THREE_PARAGRAPHS})
        source.files["blank.txt"] = "   "
        embedder = CountingEmbedder()

        count = make_indexer(source, repository, embedder).ingest(incremental=True)

        self.assertEqual(count, 3)
        self.assertEqual(embedder.calls, [])
        self.assertEqual(repository.save_calls, 1)
        self.assertEqual(repository.persisted_ids, A_CHUNK_IDS)

    # Regression: a newly added file that yields no chunks (blank file,
    # image-only PDF) must be recorded in the manifest like a full build does,
    # otherwise it is re-loaded and re-parsed on every incremental ingest.
    def test_added_file_without_chunks_is_not_reloaded_on_next_ingest(self) -> None:
        source, repository = build_index({"a.txt": THREE_PARAGRAPHS})
        source.files["blank.txt"] = "   "
        indexer = make_indexer(source, repository)
        indexer.ingest(incremental=True)
        source.loaded_paths.clear()

        indexer.ingest(incremental=True)

        self.assertIn("blank.txt", repository.manifest.files)
        self.assertEqual(source.loaded_paths, [])

    def test_added_files_without_chunks_are_logged(self) -> None:
        source, repository = build_index({"a.txt": THREE_PARAGRAPHS})
        source.files["blank.txt"] = "   "

        with self.assertLogs("rag_project.application.indexing", "WARNING") as logs:
            make_indexer(source, repository).ingest(incremental=True)

        self.assertTrue(any("blank.txt" in line for line in logs.output))

    def test_changed_file_triggers_full_rebuild(self) -> None:
        source, repository = build_index({"a.txt": THREE_PARAGRAPHS, "b.md": SHORT_NOTE})
        source.files["b.md"] = "Short note about dogs."
        embedder = CountingEmbedder()

        count = make_indexer(source, repository, embedder).ingest(incremental=True)

        self.assertEqual(count, 4)
        self.assertEqual(source.load_all_calls, 2)
        self.assertEqual(repository.new_store_calls, 2)
        self.assertEqual(len(embedder.embedded_texts), 4)
        self.assertIn("Short note about dogs.", repository.persisted_texts)
        self.assertNotIn(SHORT_NOTE, repository.persisted_texts)
        self.assertEqual(
            repository.manifest.files["b.md"].sha256, sha256_of("Short note about dogs.")
        )

    def test_deleted_file_triggers_full_rebuild(self) -> None:
        source, repository = build_index({"a.txt": THREE_PARAGRAPHS, "b.md": SHORT_NOTE})
        del source.files["b.md"]
        embedder = CountingEmbedder()

        count = make_indexer(source, repository, embedder).ingest(incremental=True)

        self.assertEqual(count, 3)
        self.assertEqual(source.load_all_calls, 2)
        self.assertEqual(repository.new_store_calls, 2)
        self.assertEqual(repository.persisted_ids, A_CHUNK_IDS)
        self.assertEqual(set(repository.manifest.files), {"a.txt"})

    def test_missing_index_with_stale_manifest_triggers_full_build(self) -> None:
        source = FakeDocumentSource({"a.txt": THREE_PARAGRAPHS})
        repository = FakeIndexRepository()
        repository.manifest = IndexManifest(
            files={
                "a.txt": SourceFileRecord(
                    path="a.txt",
                    sha256=sha256_of(THREE_PARAGRAPHS),
                    size=1,
                    mtime_ns=0,
                    chunk_ids=A_CHUNK_IDS,
                )
            }
        )

        count = make_indexer(source, repository).ingest(incremental=True)

        self.assertEqual(count, 3)
        self.assertEqual(source.load_all_calls, 1)
        self.assertEqual(repository.new_store_calls, 1)
        self.assertEqual(repository.persisted_ids, A_CHUNK_IDS)

    def test_existing_index_with_empty_manifest_triggers_full_build(self) -> None:
        source = FakeDocumentSource({"a.txt": THREE_PARAGRAPHS})
        repository = FakeIndexRepository()
        repository.persisted = [Document(id="legacy:0000", text="legacy chunk")]

        count = make_indexer(source, repository).ingest(incremental=True)

        self.assertEqual(count, 3)
        self.assertEqual(repository.new_store_calls, 1)
        self.assertEqual(repository.persisted_ids, A_CHUNK_IDS)
        self.assertEqual(set(repository.manifest.files), {"a.txt"})

    def test_full_mode_rebuilds_even_when_nothing_changed(self) -> None:
        source, repository = build_index({"a.txt": THREE_PARAGRAPHS})
        embedder = CountingEmbedder()

        count = make_indexer(source, repository, embedder).ingest(incremental=False)

        self.assertEqual(count, 3)
        self.assertEqual(source.load_all_calls, 2)
        self.assertEqual(len(embedder.embedded_texts), 3)
        self.assertEqual(repository.persisted_ids, A_CHUNK_IDS)


class RemoveDocumentTests(unittest.TestCase):
    def test_removes_only_that_files_chunks_and_updates_manifest(self) -> None:
        _, repository = build_index({"a.txt": THREE_PARAGRAPHS, "b.md": SHORT_NOTE})
        b_record = repository.manifest.files["b.md"]

        removed = remove_document(repository, "a.txt")

        self.assertEqual(removed, 3)
        self.assertEqual(repository.persisted_ids, ["b.md:0000"])
        self.assertEqual(set(repository.manifest.files), {"b.md"})
        self.assertEqual(repository.manifest.files["b.md"], b_record)

    def test_unknown_file_returns_zero_and_leaves_index_untouched(self) -> None:
        _, repository = build_index({"a.txt": THREE_PARAGRAPHS})
        manifest_before = repository.manifest

        self.assertEqual(remove_document(repository, "missing.txt"), 0)

        self.assertEqual(repository.save_calls, 1)  # only the initial build
        self.assertEqual(repository.persisted_ids, A_CHUNK_IDS)
        self.assertEqual(repository.manifest, manifest_before)

    def test_no_index_returns_zero(self) -> None:
        repository = FakeIndexRepository()

        self.assertEqual(remove_document(repository, "a.txt"), 0)

        self.assertEqual(repository.load_store_calls, 0)
        self.assertIsNone(repository.persisted)

    def test_file_without_chunks_is_dropped_from_manifest(self) -> None:
        _, repository = build_index({"a.txt": THREE_PARAGRAPHS, "empty.txt": " "})

        self.assertEqual(remove_document(repository, "empty.txt"), 0)

        self.assertEqual(set(repository.manifest.files), {"a.txt"})
        self.assertEqual(repository.persisted_ids, A_CHUNK_IDS)

    def test_removed_file_still_in_source_is_reappended_incrementally(self) -> None:
        source, repository = build_index({"a.txt": THREE_PARAGRAPHS, "b.md": SHORT_NOTE})
        remove_document(repository, "b.md")
        embedder = CountingEmbedder()

        count = make_indexer(source, repository, embedder).ingest(incremental=True)

        self.assertEqual(count, 4)
        self.assertEqual(embedder.embedded_texts, [SHORT_NOTE])
        self.assertEqual(repository.persisted_ids, [*A_CHUNK_IDS, "b.md:0000"])


class FaissIndexingIntegrationTests(unittest.TestCase):
    """The real adapters satisfy the ports the indexer depends on."""

    def test_build_append_noop_and_remove_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "documents"
            source_dir.mkdir()
            (source_dir / "a.txt").write_text(THREE_PARAGRAPHS, encoding="utf-8")
            repository = FaissIndexRepository(Path(temp_dir) / "index")
            embedder = CountingEmbedder()
            indexer = DocumentIndexer(
                embedding_model=embedder,
                sources=DocumentDirectory(source_dir),
                index=repository,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            self.assertEqual(indexer.ingest(), 3)

            (source_dir / "b.md").write_text(SHORT_NOTE, encoding="utf-8")
            embedder.calls.clear()
            self.assertEqual(indexer.ingest(), 4)
            self.assertEqual(embedder.embedded_texts, [SHORT_NOTE])

            embedder.calls.clear()
            self.assertEqual(indexer.ingest(), 4)
            self.assertEqual(embedder.calls, [])

            self.assertEqual(remove_document(repository, "a.txt"), 3)
            store = repository.load_store()
            self.assertEqual([document.id for document in store.documents], ["b.md:0000"])
            self.assertEqual(set(repository.load_manifest().files), {"b.md"})

    def test_failed_full_rebuild_keeps_previous_index(self) -> None:
        class FailingEmbedder:
            def embed_texts(self, texts):
                raise RuntimeError("model download failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "documents"
            source_dir.mkdir()
            (source_dir / "a.txt").write_text(THREE_PARAGRAPHS, encoding="utf-8")
            repository = FaissIndexRepository(Path(temp_dir) / "index")

            def indexer(embedder) -> DocumentIndexer:
                return DocumentIndexer(
                    embedding_model=embedder,
                    sources=DocumentDirectory(source_dir),
                    index=repository,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )

            self.assertEqual(indexer(CountingEmbedder()).ingest(), 3)
            (source_dir / "b.md").write_text(SHORT_NOTE, encoding="utf-8")

            with self.assertRaises(RuntimeError):
                indexer(FailingEmbedder()).ingest(incremental=False)

            self.assertEqual(repository.load_store().count(), 3)
            self.assertEqual(set(repository.load_manifest().files), {"a.txt"})
