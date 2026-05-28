"""Citation formatting for retrieved chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_project.models import SearchResult


@dataclass(frozen=True)
class Citation:
    label: str
    source: str
    page: str | None
    chunk_index: str
    score: float
    text: str


def build_citations(results: Sequence[SearchResult]) -> list[Citation]:
    citations: list[Citation] = []
    for index, result in enumerate(results, start=1):
        metadata = result.document.metadata
        source = metadata.get("relative_path") or metadata.get("source") or result.document.id
        page = metadata.get("page_label") or metadata.get("page")
        citations.append(
            Citation(
                label=f"S{index}",
                source=str(source),
                page=str(page) if page is not None else None,
                chunk_index=str(metadata.get("chunk_index", "unknown")),
                score=result.score,
                text=result.document.text,
            )
        )
    return citations


def format_citation_context(citations: Sequence[Citation]) -> list[str]:
    chunks: list[str] = []
    for citation in citations:
        page = f", page={citation.page}" if citation.page is not None else ""
        chunks.append(
            f"[{citation.label}] source={citation.source}{page}, "
            f"chunk={citation.chunk_index}\n{citation.text}"
        )
    return chunks
