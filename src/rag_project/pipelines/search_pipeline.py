"""Search an existing FAISS index by question."""

from __future__ import annotations

from pathlib import Path

from rag_project.config import Settings, load_settings
from rag_project.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel
from rag_project.models import SearchResult
from rag_project.retrieval.retriever import Retriever
from rag_project.vectorstores.faiss_store import FaissVectorStore


def search_index(
    question: str,
    index_dir: str | Path | None = None,
    top_k: int = 5,
    settings: Settings | None = None,
) -> list[SearchResult]:
    active_settings = settings or load_settings()
    store = FaissVectorStore.load_from_disk(index_dir or active_settings.vector_store_dir)

    retriever = Retriever(
        embedding_model=SentenceTransformerEmbeddingModel(active_settings.embedding_model),
        vector_store=store,
    )
    return retriever.search(question, top_k=top_k)
