"""Vector store abstractions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from rag_project.models import Document, SearchResult


class VectorStore(Protocol):
    def add(self, documents: Sequence[Document], embeddings: Sequence[Sequence[float]]) -> None:
        """Add documents and their embeddings to the store."""

    def search(self, embedding: Sequence[float], top_k: int = 5) -> list[SearchResult]:
        """Return the nearest documents for an embedding."""

    def count(self) -> int:
        """Return the number of stored documents."""

    def remove_ids(self, document_ids: Sequence[str]) -> int:
        """Remove documents by their IDs; return the number removed."""


class InMemoryVectorStore:
    """Simple vector store for tests and local prototypes."""

    def __init__(self) -> None:
        self._items: list[tuple[Document, list[float]]] = []

    def add(self, documents: Sequence[Document], embeddings: Sequence[Sequence[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")

        for document, embedding in zip(documents, embeddings, strict=True):
            self._items.append((document, list(embedding)))

    def count(self) -> int:
        return len(self._items)

    def remove_ids(self, document_ids: Sequence[str]) -> int:
        remove_set = set(document_ids)
        before = len(self._items)
        self._items = [
            (document, embedding)
            for document, embedding in self._items
            if document.id not in remove_set
        ]
        return before - len(self._items)

    def search(self, embedding: Sequence[float], top_k: int = 5) -> list[SearchResult]:
        scored = [
            SearchResult(document=document, score=_cosine_similarity(embedding, stored_embedding))
            for document, stored_embedding in self._items
        ]
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
