"""JSON-friendly (de)serialization of domain entities for file storage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rag_project.domain.models import Document, SearchResult


def document_to_dict(document: Document) -> dict[str, Any]:
    return {"id": document.id, "text": document.text, "metadata": document.metadata}


def document_from_dict(payload: Mapping[str, Any]) -> Document:
    return Document(
        id=payload["id"],
        text=payload["text"],
        metadata=payload.get("metadata", {}),
    )


def search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    return {"document": document_to_dict(result.document), "score": result.score}


def search_result_from_dict(payload: Mapping[str, Any]) -> SearchResult:
    return SearchResult(
        document=document_from_dict(payload["document"]),
        score=payload.get("score", 0.0),
    )
