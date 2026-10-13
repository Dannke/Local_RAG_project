"""Semantic retrieval: an embedding model joined with a vector store."""

from __future__ import annotations

from collections.abc import Sequence

from rag_project.application.ports import EmbeddingModel, VectorStore
from rag_project.domain.models import Document, SearchResult


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def index(self, documents: Sequence[Document]) -> None:
        embeddings = self.embedding_model.embed_texts([document.text for document in documents])
        self.vector_store.add(documents, embeddings)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        embedding = self.embedding_model.embed_texts([query])[0]
        return self.vector_store.search(embedding, top_k=top_k)


class NoOpReranker:
    """Keep the retriever's order; used when cross-encoder reranking is off."""

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        return list(results)[:top_k]
