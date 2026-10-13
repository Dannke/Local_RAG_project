"""Right panel of the Chat page: citation explorer for the last answer.

Pure UI — reads already-retrieved results and renders them as cards. It never
touches retrieval, reranking, scoring or citation logic.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

import streamlit as st

from rag_project.domain.citations import Citation, build_citations
from rag_project.domain.models import SearchResult
from rag_project.ui.formatting import (
    citation_filename,
    citation_preview,
    file_icon,
    file_type,
    relevance_for,
)
from rag_project.ui.theme import render_html


def render_sources_panel(results: Sequence[SearchResult]) -> None:
    citations = build_citations(results)

    _render_header(len(citations))

    if not citations:
        _render_empty()
        return

    for index, citation in enumerate(citations, start=1):
        _render_citation_card(index, citation)


def _render_header(total: int) -> None:
    render_html(
        '<div class="citation-panel-header">'
        '<h3 class="citation-panel-title">📚 Sources</h3>'
        f'<p class="citation-panel-subtitle">Total Sources: <b>{total}</b></p>'
        "</div>"
    )


def _render_empty() -> None:
    render_html(
        '<div class="citation-empty">'
        '<div class="citation-empty-icon">📚</div>'
        '<div class="citation-empty-title">No sources available.</div>'
        '<div class="citation-empty-text">'
        "Ask a question to view retrieved citations."
        "</div>"
        "</div>"
    )


def _render_citation_card(index: int, citation: Citation) -> None:
    """A single citation card.

    Layout: source number + filename, file type / page / chunk chips, an
    optional relevance bar, a chunk preview and an expander with the full text.
    """
    filename = citation_filename(citation.source)
    extension = Path(filename).suffix.lower()

    chips = [f'<span class="citation-chip"><b>{file_type(extension)}</b></span>']
    if citation.page:
        chips.append(f'<span class="citation-chip">📄 <b>Page:</b> {citation.page}</span>')
    chips.append(f'<span class="citation-chip"><b>Chunk:</b> {citation.chunk_index}</span>')

    render_html(
        '<div class="citation-card">'
        '<div class="citation-card-head">'
        f'<span class="citation-source-badge">Source #{index}</span>'
        f'<span class="citation-source-name">{file_icon(extension)} '
        f"{html.escape(filename)}</span>"
        "</div>"
        f'<div class="citation-meta-row">{"".join(chips)}</div>'
        f"{_relevance_html(citation.score)}"
        f'<div class="citation-preview">{html.escape(citation_preview(citation.text))}</div>'
        "</div>"
    )

    with st.expander("Show Full Chunk"):
        st.write(citation.text)


def _relevance_html(score: float) -> str:
    """Relevance bar, or an empty string when the score is unusable."""
    relevance = relevance_for(score)
    if relevance is None:
        return ""
    percent, label, color = relevance
    return (
        '<div class="citation-relevance">'
        '<span class="citation-relevance-label">Relevance</span>'
        '<span class="citation-relevance-bar">'
        '<span class="citation-relevance-fill" '
        f'style="width:{percent}%;background:{color};"></span>'
        "</span>"
        f'<span class="citation-relevance-value" style="color:{color};">'
        f"{percent}% · {label}</span>"
        "</div>"
    )
