"""Question answering use case: retrieve, rerank, cite, generate."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from rag_project.application.observability import new_request_id, now_ms
from rag_project.application.ports import AnswerGenerator, Reranker
from rag_project.application.retrieval import NoOpReranker, Retriever
from rag_project.domain.citations import Citation, build_citations, format_citation_context
from rag_project.domain.models import Document, SearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    results: list[SearchResult]
    citations: list[Citation]

    @property
    def contexts(self) -> list[Document]:
        return [result.document for result in self.results]


@dataclass(frozen=True)
class ChatStreamResponse:
    chunks: Iterator[str]
    results: list[SearchResult]
    citations: list[Citation]

    @property
    def contexts(self) -> list[Document]:
        return [result.document for result in self.results]


class ChatSession:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        default_top_k: int = 5,
        reranker: Reranker | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.default_top_k = default_top_k
        self.reranker = reranker or NoOpReranker()
        self.rerank_candidates = rerank_candidates

    def ask(self, question: str, top_k: int | None = None) -> ChatResponse:
        results, citations = self._prepare(question, top_k, event="answer_request")
        return ChatResponse(
            answer=self.generator.generate(question, format_citation_context(citations)),
            results=results,
            citations=citations,
        )

    def stream(self, question: str, top_k: int | None = None) -> ChatStreamResponse:
        results, citations = self._prepare(question, top_k, event="stream_request")
        return ChatStreamResponse(
            chunks=self.generator.stream(question, format_citation_context(citations)),
            results=results,
            citations=citations,
        )

    def _prepare(
        self, question: str, top_k: int | None, event: str
    ) -> tuple[list[SearchResult], list[Citation]]:
        """Start a request: retrieve, rerank and build the citations."""
        new_request_id()
        final_top_k = top_k or self.default_top_k
        results = self._search(question, final_top_k)
        citations = build_citations(results)
        logger.info(
            event,
            extra={
                "question": question,
                "final_top_k": final_top_k,
                "retrieved": len(results),
            },
        )
        return results, citations

    def _search(self, question: str, final_top_k: int) -> list[SearchResult]:
        candidate_count = max(final_top_k, self.rerank_candidates)
        start = now_ms()
        candidates = self.retriever.search(question, top_k=candidate_count)
        retrieval_ms = now_ms() - start
        reranked = self.reranker.rerank(question, candidates, top_k=final_top_k)
        logger.info(
            "retrieval",
            extra={
                "question": question,
                "candidates_before_rerank": len(candidates),
                "final_top_k": final_top_k,
                "retrieved": len(reranked),
                "retrieval_ms": retrieval_ms,
                "top_scores": [round(result.score, 4) for result in reranked[:5]],
            },
        )
        return reranked
