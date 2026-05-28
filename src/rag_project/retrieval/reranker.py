"""Reranking utilities for retrieved chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_project.models import SearchResult


class Reranker(Protocol):
    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        """Return reranked search results."""


class NoOpReranker:
    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        return list(results)[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers to use CrossEncoder reranking."
            ) from exc

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, result.document.text) for result in results]
        scores = self.model.predict(pairs)
        reranked = [
            SearchResult(document=result.document, score=float(score))
            for result, score in zip(results, scores, strict=True)
        ]
        return sorted(reranked, key=lambda result: result.score, reverse=True)[:top_k]
