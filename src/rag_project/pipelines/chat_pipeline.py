"""Interactive RAG chat pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from rag_project.config import Settings, load_settings
from rag_project.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel
from rag_project.generation.llm import LLM
from rag_project.llm.llm_client import OpenRouterGenerator
from rag_project.models import Document, SearchResult
from rag_project.retrieval.citations import Citation, build_citations, format_citation_context
from rag_project.retrieval.reranker import CrossEncoderReranker, NoOpReranker, Reranker
from rag_project.retrieval.retriever import Retriever
from rag_project.vectorstores.faiss_store import FaissVectorStore


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    contexts: list[Document]
    results: list[SearchResult]
    citations: list[Citation]


@dataclass(frozen=True)
class ChatStreamResponse:
    chunks: Iterator[str]
    contexts: list[Document]
    results: list[SearchResult]
    citations: list[Citation]


class ChatSession:
    def __init__(
        self,
        retriever: Retriever,
        generator: LLM | None = None,
        default_top_k: int = 5,
        reranker: Reranker | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or OpenRouterGenerator()
        self.default_top_k = default_top_k
        self.reranker = reranker or NoOpReranker()
        self.rerank_candidates = rerank_candidates

    @classmethod
    def from_faiss_index(
        cls,
        index_dir: str | Path | None = None,
        settings: Settings | None = None,
        generator: LLM | None = None,
    ) -> ChatSession:
        active_settings = settings or load_settings()
        store = FaissVectorStore.load_from_disk(index_dir or active_settings.vector_store_dir)
        retriever = Retriever(
            embedding_model=SentenceTransformerEmbeddingModel(active_settings.embedding_model),
            vector_store=store,
        )
        reranker: Reranker
        if getattr(active_settings, 'use_reranker', True):
            reranker_model = getattr(
                active_settings,
                'reranker_model',
                'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
            )
            reranker = CrossEncoderReranker(reranker_model)
        else:
            reranker = NoOpReranker()
        return cls(
            retriever=retriever,
            generator=generator or OpenRouterGenerator(settings=active_settings),
            default_top_k=active_settings.top_k,
            reranker=reranker,
            rerank_candidates=getattr(active_settings, 'rerank_candidates', 20),
        )

    def ask(self, question: str, top_k: int | None = None) -> ChatResponse:
        results = self._search(question, top_k)
        contexts = [result.document for result in results]
        citations = build_citations(results)
        context_chunks = format_citation_context(citations)
        return ChatResponse(
            answer=self.generator.generate(question, context_chunks),
            contexts=contexts,
            results=results,
            citations=citations,
        )

    def stream(self, question: str, top_k: int | None = None) -> ChatStreamResponse:
        results = self._search(question, top_k)
        contexts = [result.document for result in results]
        citations = build_citations(results)
        context_chunks = format_citation_context(citations)
        return ChatStreamResponse(
            chunks=self.generator.stream(question, context_chunks),
            contexts=contexts,
            results=results,
            citations=citations,
        )

    def _search(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        final_top_k = top_k or self.default_top_k
        candidate_count = max(final_top_k, self.rerank_candidates)
        candidates = self.retriever.search(question, top_k=candidate_count)
        return self.reranker.rerank(question, candidates, top_k=final_top_k)
