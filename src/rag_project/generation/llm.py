"""LLM interface and context-only fallback implementation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol


class LLM(Protocol):
    def generate(self, question: str, contexts: Sequence[str]) -> str:
        """Generate an answer from a question and retrieved contexts."""

    def stream(self, question: str, contexts: Sequence[str]) -> Iterator[str]:
        """Stream an answer from a question and retrieved contexts."""


class ContextOnlyGenerator:
    """Return retrieved context without calling an LLM."""

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        if not contexts:
            return "No relevant context was found."

        context_text = "\n\n".join(
            f"[{index}] {text}" for index, text in enumerate(contexts, start=1)
        )
        return f"Question: {question}\n\nRetrieved context:\n{context_text}"

    def stream(self, question: str, contexts: Sequence[str]) -> Iterator[str]:
        yield self.generate(question, contexts)
