"""Composition root: builds use cases with their adapters from ``Settings``.

Entry points (Streamlit UI, CLI, scripts, eval) get chat sessions, retrievers,
indexers and LLM clients from these factories, so swapping a model or provider
is a change here only. The UI additionally works with the per-chat storage
adapters (``ui.workspace``) for plain file management.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag_project.application.chat import ChatSession
from rag_project.application.indexing import DocumentIndexer, ProgressCallback
from rag_project.application.observability import now_ms
from rag_project.application.ports import AnswerGenerator, Reranker
from rag_project.application.retrieval import NoOpReranker, Retriever
from rag_project.config import Settings, load_settings
from rag_project.domain.models import SearchResult
from rag_project.infrastructure.document_directory import DocumentDirectory
from rag_project.infrastructure.embeddings import SentenceTransformerEmbeddingModel
from rag_project.infrastructure.faiss_store import FaissVectorStore
from rag_project.infrastructure.index_storage import FaissIndexRepository
from rag_project.infrastructure.openrouter import OpenRouterClient, OpenRouterGenerator
from rag_project.infrastructure.rate_limit import RateLimiter
from rag_project.infrastructure.rerankers import CrossEncoderReranker

logger = logging.getLogger(__name__)


def create_llm_client(
    settings: Settings | None = None,
    rate_limit_key: str | None = None,
) -> OpenRouterClient:
    active = settings or load_settings()
    rate_limiter = None
    if active.rate_limit_max_requests > 0:
        rate_limiter = RateLimiter(
            max_requests=active.rate_limit_max_requests,
            window_seconds=active.rate_limit_window_seconds,
        )
    return OpenRouterClient(
        api_key=active.openrouter_api_key or "",
        model=active.openrouter_model,
        base_url=active.openrouter_base_url,
        timeout_seconds=active.openrouter_timeout_seconds,
        temperature=active.temperature,
        max_context_chars=active.max_context_chars,
        rate_limiter=rate_limiter,
        rate_limit_key=rate_limit_key,
    )


def create_retriever(
    index_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> Retriever:
    """Retriever over a saved FAISS index; raises ``FileNotFoundError`` if missing."""
    active = settings or load_settings()
    return Retriever(
        embedding_model=SentenceTransformerEmbeddingModel(active.embedding_model),
        vector_store=FaissVectorStore.load_from_disk(index_dir or active.vector_store_dir),
    )


def create_chat_session(
    index_dir: str | Path | None = None,
    settings: Settings | None = None,
    generator: AnswerGenerator | None = None,
    rate_limit_key: str | None = None,
) -> ChatSession:
    active = settings or load_settings()
    retriever = create_retriever(index_dir, active)
    reranker: Reranker = (
        CrossEncoderReranker(active.reranker_model) if active.use_reranker else NoOpReranker()
    )
    return ChatSession(
        retriever=retriever,
        generator=generator
        or OpenRouterGenerator(lambda: create_llm_client(active, rate_limit_key)),
        default_top_k=active.top_k,
        reranker=reranker,
        rerank_candidates=active.rerank_candidates,
    )


def create_indexer(
    source_dir: str | Path,
    index_dir: str | Path,
    settings: Settings,
) -> DocumentIndexer:
    embedding_model = SentenceTransformerEmbeddingModel(settings.embedding_model)
    return DocumentIndexer(
        embedding_model=embedding_model,
        sources=DocumentDirectory(source_dir),
        index=FaissIndexRepository(index_dir),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        token_counter=embedding_model.count_tokens,
    )


def ingest_to_faiss(
    input_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    settings: Settings | None = None,
    incremental: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> int:
    """Index ``input_dir`` into a FAISS index; return the index chunk count.

    ``progress_callback`` is called as ``progress_callback(stage, current, total)``.
    """
    active = settings or load_settings()
    source_dir = Path(input_dir) if input_dir is not None else active.raw_data_dir
    target_dir = Path(index_dir) if index_dir is not None else active.vector_store_dir

    start = now_ms()
    logger.info(
        "ingest_start",
        extra={
            "source_dir": str(source_dir),
            "index_dir": str(target_dir),
            "incremental": incremental,
        },
    )
    try:
        count = create_indexer(source_dir, target_dir, active).ingest(
            incremental=incremental, progress=progress_callback
        )
    except Exception as exc:
        logger.error(
            "ingest_error",
            extra={
                "error": exc.__class__.__name__,
                "error_message": str(exc),
                "elapsed_ms": now_ms() - start,
            },
            exc_info=True,
        )
        raise
    logger.info(
        "ingest_completed",
        extra={"chunks": count, "elapsed_ms": now_ms() - start, "success": True},
    )
    return count


def search_index(
    question: str,
    index_dir: str | Path | None = None,
    top_k: int = 5,
    settings: Settings | None = None,
) -> list[SearchResult]:
    return create_retriever(index_dir, settings).search(question, top_k=top_k)
