"""Core entities shared across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RELATIVE_PATH_KEY = "relative_path"
"""Metadata key linking a document or chunk to its source file (the manifest key)."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def relative_path(self) -> str | None:
        """Source file path relative to the documents root, if known."""
        value = self.metadata.get(RELATIVE_PATH_KEY)
        return str(value) if value else None


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
