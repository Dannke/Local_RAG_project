"""Indexing use cases: keep a vector index in sync with its source documents."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping

from rag_project.application.ports import (
    DocumentSource,
    EmbeddingModel,
    IndexRepository,
    PersistentVectorStore,
)
from rag_project.domain.chunking import TokenCounter, chunk_documents
from rag_project.domain.errors import NoDocumentsToIndexError
from rag_project.domain.manifest import IndexManifest, SourceFileRecord, chunk_ids_by_source
from rag_project.domain.models import Document

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]
"""Called as ``progress(stage_name, current, total)``."""


def _ignore_progress(stage: str, current: int, total: int) -> None:
    pass


class DocumentIndexer:
    """Build or incrementally update one index from one document source.

    Incremental mode appends only newly added files. A changed or deleted
    source file forces a full rebuild, because the FAISS index type in use
    does not support reliable in-place vector updates.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        sources: DocumentSource,
        index: IndexRepository,
        chunk_size: int,
        chunk_overlap: int,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._sources = sources
        self._index = index
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._token_counter = token_counter

    def ingest(self, incremental: bool = True, progress: ProgressCallback | None = None) -> int:
        """Index the source documents; return the number of chunks in the index."""
        report = progress or _ignore_progress
        records = self._sources.scan()
        if incremental:
            return self._ingest_incremental(records, report)
        return self._ingest_full(records, report)

    def _ingest_full(
        self, records: Mapping[str, SourceFileRecord], report: ProgressCallback
    ) -> int:
        report("Загрузка документов", 0, 1)
        documents = self._sources.load_all()

        report("Разделение на фрагменты", 0, 1)
        chunks = self._chunk(documents)
        if not chunks:
            raise NoDocumentsToIndexError("No supported documents were found to index.")

        report("Инициализация индекса", 0, 1)
        store = self._index.new_store()

        report("Индексирование фрагментов", 0, len(chunks))
        self._embed_into(store, chunks)

        report("Сохранение индекса", 0, 1)
        store.save()
        _warn_sources_without_chunks(records, chunks)
        self._index.save_manifest(IndexManifest.from_chunks(records, chunks))
        return store.count()

    def _ingest_incremental(
        self, records: Mapping[str, SourceFileRecord], report: ProgressCallback
    ) -> int:
        manifest = self._index.load_manifest()
        if not self._index.exists() or not manifest.files:
            return self._ingest_full(records, report)

        diff = manifest.diff(records)
        if diff.requires_rebuild:
            return self._ingest_full(records, report)
        if not diff.added:
            return self._index.load_store().count()

        total = len(diff.added)
        report("Загрузка новых документов", 0, total)
        documents: list[Document] = []
        for loaded, relative_path in enumerate(diff.added, start=1):
            documents.extend(self._sources.load(relative_path))
            report("Загрузка новых документов", loaded, total)

        report("Разделение на фрагменты", 0, 1)
        chunks = self._chunk(documents)
        _warn_sources_without_chunks(diff.added, chunks)
        if not chunks:
            self._index.save_manifest(manifest.with_appended(records, diff.added, chunks))
            return self._index.load_store().count()

        report("Загрузка существующего индекса", 0, 1)
        store = self._index.load_store()

        report("Добавление новых фрагментов", 0, len(chunks))
        self._embed_into(store, chunks)

        report("Сохранение индекса", 0, 1)
        store.save()
        self._index.save_manifest(manifest.with_appended(records, diff.added, chunks))
        return store.count()

    def _chunk(self, documents: list[Document]) -> list[Document]:
        return chunk_documents(
            documents,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            token_counter=self._token_counter,
        )

    def _embed_into(self, store: PersistentVectorStore, chunks: list[Document]) -> None:
        store.add(chunks, self._embedding_model.embed_texts([chunk.text for chunk in chunks]))


def _warn_sources_without_chunks(paths: Iterable[str], chunks: list[Document]) -> None:
    """Log source files that yielded no text (e.g. image-only PDFs)."""
    chunked = chunk_ids_by_source(chunks)
    empty = sorted(path for path in paths if path not in chunked)
    if empty:
        logger.warning(
            "sources_without_chunks: %s", ", ".join(empty), extra={"sources": empty}
        )


def remove_document(index: IndexRepository, relative_path: str) -> int:
    """Remove one source document's chunks from an existing index.

    The other documents stay untouched (no full rebuild). Chunk ids come from
    the manifest; if it has no record for the file (unreadable manifest, or an
    index older than manifests) they are found by the chunks' own metadata, so
    a deleted file's vectors are never left behind. Returns the number of
    chunks removed: 0 when there is no index or the document is not indexed.
    """
    if not index.exists():
        return 0

    manifest = index.load_manifest()
    record = manifest.files.get(relative_path)
    store = index.load_store()
    if record is not None:
        chunk_ids = record.chunk_ids
    else:
        chunk_ids = store.ids_for_source(relative_path)
        if chunk_ids:
            logger.warning(
                "manifest_record_missing: %s; removing %d chunks by metadata",
                relative_path,
                len(chunk_ids),
            )

    removed = store.remove_ids(chunk_ids) if chunk_ids else 0
    if removed:
        store.save()
    if record is not None:
        index.save_manifest(manifest.without(relative_path))
    return removed
