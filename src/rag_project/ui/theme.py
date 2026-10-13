"""Visual building blocks: stylesheet, avatars and shared HTML fragments.

Every fragment here is rendered with ``unsafe_allow_html``: callers pass
already-escaped text (``html.escape``) for anything that comes from the user.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_project.config import Settings

USER_AVATAR = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "viewBox='0 0 24 24' fill='none' stroke='%23A9AFB8' stroke-width='1.7' "
    "stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 "
    "4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/%3E%3Ccircle cx='12' cy='7' r='4'/%3E%3C/svg%3E"
)
# An open book: answers come only from the chat's own documents.
ASSISTANT_AVATAR = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "viewBox='0 0 24 24' fill='none' stroke='%23E2A54B' stroke-width='1.7' "
    "stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 7v14'/%3E"
    "%3Cpath d='M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 "
    "4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z'/%3E%3C/svg%3E"
)

TYPING_INDICATOR_HTML = (
    '<div class="typing" role="status">'
    '<span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span>'
    '<span class="typing-label">Ищу контекст, ранжирую источники и запрашиваю модель…</span>'
    "</div>"
)


def inject_css(style_path: Path) -> None:
    css = style_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def chat_avatar(role: str) -> str:
    return USER_AVATAR if role == "user" else ASSISTANT_AVATAR


def render_html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = "") -> None:
    """Page title with an optional muted subtitle (both pre-escaped)."""
    subtitle_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    render_html(f'<header class="page-header"><h1>{title}</h1>{subtitle_html}</header>')


def render_section_title(title: str, hint: str = "") -> None:
    hint_html = f'<span class="section-hint">{hint}</span>' if hint else ""
    render_html(f'<div class="section-title"><h2>{title}</h2>{hint_html}</div>')


def empty_state_html(title: str, text: str, *, mark: str = "") -> str:
    mark_html = f'<div class="empty-mark" aria-hidden="true">{mark}</div>' if mark else ""
    return (
        f'<div class="empty-state">{mark_html}'
        f'<p class="empty-title">{title}</p>'
        f'<p class="empty-text">{text}</p></div>'
    )


def render_openrouter_warning(settings: Settings) -> None:
    if settings.openrouter_api_key:
        return
    render_html(
        '<div class="callout callout-danger" role="alert">'
        "<b>OPENROUTER_API_KEY</b> не найден. Добавьте ключ в <code>.env</code> "
        "перед запуском чата.</div>"
    )


def kpi_card_html(label: str, value: object, *, tone: str = "", note: str = "") -> str:
    """A dashboard figure; ``tone`` is a CSS modifier (ok / warn)."""
    tone_class = f" kpi-{tone}" if tone else ""
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    return (
        f'<div class="kpi{tone_class}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{note_html}</div>'
    )
