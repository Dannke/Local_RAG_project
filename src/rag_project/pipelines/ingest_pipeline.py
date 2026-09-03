"""Index documents into a vector store."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from rag_project.config import Settings, load_settings
from rag_project.embeddings.base import EmbeddingModel
from rag_project.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel
from rag_project.ingestion.chunking import chunk_documents
from rag_project.ingestion.loaders import load_document_file, load_documents
from rag_project.models import Document
from rag_project.pipelines.index_manifest import (
    IndexManifest,
    SourceFileRecord,
    scan_source_files,
    with_chunk_ids,
)
from rag_project.retrieval.retriever import Retriever
from rag_project.vectorstores.base import VectorStore
from rag_project.vectorstores.faiss_store import FaissVectorStore


def build_retriever(
    input_dir: str | Path | None = None,
    settings: Settings | None = None,
    embedding_model: EmbeddingModel | None = None,
    vector_store: VectorStore | None = None,
) -> Retriever:
    active_settings = settings or load_settings()
    source_dir = Path(input_dir) if input_dir is not None else active_settings.raw_data_dir

    documents = load_documents(source_dir)
    chunks = _chunk_loaded_documents(documents, active_settings)

    retriever = Retriever(
        embedding_model=embedding_model
        or SentenceTransformerEmbeddingModel(active_settings.embedding_model),
        vector_store=vector_store or FaissVectorStore(active_settings.vector_store_dir),
    )
    retriever.index(chunks)
    return retriever


def ingest_to_faiss(
    input_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    settings: Settings | None = None,
    incremental: bool = True,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> int:
    """Ingest documents to FAISS index.
    
    Args:
        input_dir: Directory with source documents
        index_dir: Directory for FAISS index
        settings: Configuration settings
        incremental: Whether to do incremental indexing
        progress_callback: Optional callback for progress updates.
            Called as progress_callback(stage_name, current, total)
    """
    active_settings = settings or load_settings()
    source_dir = Path(input_dir) if input_dir is not None else active_settings.raw_data_dir
    target_index_dir = (
        Path(index_dir) if index_dir is not None else active_settings.vector_store_dir
    )

    if incremental:
        return _ingest_incremental(source_dir, target_index_dir, active_settings, progress_callback)
    return _ingest_full(source_dir, target_index_dir, active_settings, progress_callback)


def _ingest_full(
    source_dir: Path,
    index_dir: Path,
    settings: Settings,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> int:
    if progress_callback:
        progress_callback("Загрузка документов", 0, 1)
    
    documents = load_documents(source_dir)
    
    if progress_callback:
        progress_callback("Разделение на фрагменты", 0, 1)
    
    chunks = _chunk_loaded_documents(documents, settings)
    if not chunks:
        raise RuntimeError("No supported documents were found to index.")

    if progress_callback:
        progress_callback("Инициализация индекса", 0, 1)
    
    store = FaissVectorStore(index_dir)
    retriever = Retriever(
        embedding_model=SentenceTransformerEmbeddingModel(settings.embedding_model),
        vector_store=store,
    )
    
    if progress_callback:
        progress_callback("Индексирование фрагментов", 0, len(chunks))
    
    retriever.index(chunks)
    
    if progress_callback:
        progress_callback("Сохранение индекса", 0, 1)
    
    store.save_to_disk()
    _save_manifest_for_chunks(source_dir, index_dir, chunks)
    return store.count()


def _ingest_incremental(
    source_dir: Path,
    index_dir: Path,
    settings: Settings,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> int:
    current_records = scan_source_files(source_dir)
    old_manifest = IndexManifest.load(index_dir)

    if not _has_existing_index(index_dir) or not old_manifest.files:
        return _ingest_full(source_dir, index_dir, settings, progress_callback)

    deleted = set(old_manifest.files) - set(current_records)
    changed = [
        path
        for path, record in current_records.items()
        if path in old_manifest.files and record.sha256 != old_manifest.files[path].sha256
    ]
    if deleted or changed:
        return _ingest_full(source_dir, index_dir, settings, progress_callback)

    new_paths = [path for path in current_records if path not in old_manifest.files]
    if not new_paths:
        return FaissVectorStore.load_from_disk(index_dir).count()

    if progress_callback:
        progress_callback("Загрузка новых документов", 0, len(new_paths))
    
    documents: list[Document] = []
    for i, relative_path in enumerate(new_paths):
        documents.extend(load_document_file(source_dir / relative_path, root=source_dir))
        if progress_callback:
            progress_callback("Загрузка новых документов", i + 1, len(new_paths))
    
    if progress_callback:
        progress_callback("Разделение на фрагменты", 0, 1)
    
    chunks = _chunk_loaded_documents(documents, settings)
    if not chunks:
        return FaissVectorStore.load_from_disk(index_dir).count()

    if progress_callback:
        progress_callback("Загрузка существующего индекса", 0, 1)
    
    store = FaissVectorStore.load_from_disk(index_dir)
    retriever = Retriever(
        embedding_model=SentenceTransformerEmbeddingModel(settings.embedding_model),
        vector_store=store,
    )
    
    if progress_callback:
        progress_callback("Добавление новых фрагментов", 0, len(chunks))
    
    retriever.index(chunks)
    
    if progress_callback:
        progress_callback("Сохранение индекса", 0, 1)
    
    store.save_to_disk()

    updated_files = dict(old_manifest.files)
    for path, chunk_ids in _chunk_ids_by_source(chunks).items():
        updated_files[path] = with_chunk_ids(current_records[path], chunk_ids)
    IndexManifest(files=updated_files).save(index_dir)
    return store.count()


def _chunk_loaded_documents(documents: list[Document], settings: Settings) -> list[Document]:
    return chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def _save_manifest_for_chunks(source_dir: Path, index_dir: Path, chunks: list[Document]) -> None:
    current_records = scan_source_files(source_dir)
    chunk_ids_by_source = _chunk_ids_by_source(chunks)
    files: dict[str, SourceFileRecord] = {}

    for path, record in current_records.items():
        files[path] = with_chunk_ids(record, chunk_ids_by_source.get(path, []))

    IndexManifest(files=files).save(index_dir)


def _chunk_ids_by_source(chunks: list[Document]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        relative_path = chunk.metadata.get("relative_path")
        if relative_path:
            grouped[str(relative_path)].append(chunk.id)
    return dict(grouped)


def _has_existing_index(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "documents.json").exists()
