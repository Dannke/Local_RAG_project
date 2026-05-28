"""RAG query pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from rag_project.generation.llm import LLM, ContextOnlyGenerator
from rag_project.models import Document
from rag_project.retrieval.retriever import Retriever


@dataclass(frozen=True)
class RagResponse:
    answer: str
    contexts: list[Document]


class RagPipeline:
    def __init__(self, retriever: Retriever, generator: LLM | None = None) -> None:
        self.retriever = retriever
        self.generator = generator or ContextOnlyGenerator()

    def answer(self, question: str, top_k: int = 5) -> RagResponse:
        results = self.retriever.search(question, top_k=top_k)
        contexts = [result.document for result in results]
        answer = self.generator.generate(question, contexts)
        return RagResponse(answer=answer, contexts=contexts)
