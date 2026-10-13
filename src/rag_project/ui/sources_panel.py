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
    file_type,
    plural_ru,
    relevance_for,
    relevance_tier,
)
from rag_project.ui.theme import empty_state_html, render_html


def render_sources_panel(results: Sequence[SearchResult]) -> None:
    citations = build_citations(results)
    # Relevance is shown relative to the other sources of the same answer.
    scores = [citation.score for citation in citations if citation.score is not None]
    score_range = (min(scores), max(scores)) if scores else (None, None)

    _render_header(len(citations))

    if not citations:
        _render_empty()
        return

    for index, citation in enumerate(citations, start=1):
        _render_citation_card(index, citation, score_range)


def _render_header(total: int) -> None:
    count = plural_ru(total, "фрагмент", "фрагмента", "фрагментов") if total else "пока пусто"
    render_html(
        '<div class="sources-header">'
        '<h2 class="sources-title">Источники</h2>'
        f'<span class="sources-count">{count}</span>'
        "</div>"
    )


def _render_empty() -> None:
    render_html(
        empty_state_html(
            "Здесь будут цитаты",
            "После ответа появятся фрагменты документов, на которых он основан.",
        )
    )


def _render_citation_card(
    index: int,
    citation: Citation,
    score_range: tuple[float | None, float | None],
) -> None:
    """A single citation card.

    Layout: the label used in the answer ([S1]…), the filename, file type / page /
    chunk metadata, an optional relevance meter, a preview and an expander
    with the full chunk text.
    """
    filename = citation_filename(citation.source)
    label = html.escape(citation.label)
    extension = Path(filename).suffix.lower()

    meta = [f"<span>{file_type(extension)}</span>"]
    if citation.page:
        meta.append(f"<span>стр. {html.escape(str(citation.page))}</span>")
    meta.append(f"<span>фрагмент {html.escape(str(citation.chunk_index))}</span>")

    # --i staggers the entrance animation of consecutive cards.
    render_html(
        f'<article class="citation-card" style="--i:{index - 1}">'
        '<div class="citation-head">'
        f'<span class="citation-tag" title="Ссылка в ответе: [{label}]">{label}</span>'
        f'<span class="citation-name">{html.escape(filename)}</span>'
        "</div>"
        f'<div class="citation-meta">{"".join(meta)}</div>'
        f"{_relevance_html(citation.score, score_range)}"
        f'<p class="citation-preview">{html.escape(citation_preview(citation.text))}</p>'
        "</article>"
    )

    with st.expander("Полный текст фрагмента", icon=":material/notes:"):
        # PDF extraction leaves line breaks between words; read it as one flow.
        full_text = html.escape(" ".join(citation.text.split()))
        render_html(f'<p class="citation-full">{full_text}</p>')


def _relevance_html(score: float, score_range: tuple[float | None, float | None]) -> str:
    """Relevance meter, or an empty string when the score is unusable."""
    relevance = relevance_for(score, *score_range)
    if relevance is None:
        return ""
    percent, label, _ = relevance
    tier, tier_label = relevance_tier(label)
    return (
        f'<div class="relevance relevance-{tier}">'
        f'<span class="relevance-track" role="meter" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{percent}" aria-label="Релевантность {percent}%">'
        f'<span class="relevance-fill" style="--p:{percent / 100:.2f}"></span>'
        "</span>"
        f'<span class="relevance-value">{percent}% · {tier_label}</span>'
        "</div>"
    )
