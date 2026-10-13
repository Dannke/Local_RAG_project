"""Visual building blocks: stylesheet, avatars and shared HTML fragments."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_project.config import Settings

USER_AVATAR = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "viewBox='0 0 24 24' fill='none' stroke='%23d9d9d9' stroke-width='1.7' "
    "stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 "
    "4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/%3E%3Ccircle cx='12' cy='7' r='4'/%3E%3C/svg%3E"
)
ASSISTANT_AVATAR = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "viewBox='0 0 24 24' fill='none' stroke='%23d9d9d9' stroke-width='1.7' "
    "stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='11' "
    "width='18' height='10' rx='2'/%3E%3Ccircle cx='12' cy='5' r='2'/%3E%3Cpath "
    "d='M12 7v4'/%3E%3Cpath d='M8 16h.01'/%3E%3Cpath d='M16 16h.01'/%3E%3C/svg%3E"
)

ACCENT_COLOR = "#7c4dff"
SUCCESS_COLOR = "#66bb6a"
WARNING_COLOR = "#ffa726"


def inject_css(style_path: Path) -> None:
    css = style_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def chat_avatar(role: str) -> str:
    return USER_AVATAR if role == "user" else ASSISTANT_AVATAR


def render_html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def render_app_header() -> None:
    render_html(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="margin:0; font-size:1.8rem; font-weight:700;">
                🧬 <span style="background:linear-gradient(135deg,#7C4DFF,#448AF5);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                Local RAG</span>
            </h1>
            <p style="color:#8b92a8;margin:0;font-size:0.85rem;">
                Загрузка документов · FAISS-индексация · Reranking · OpenRouter LLM
            </p>
        </div>
        """
    )


def render_section_divider() -> None:
    render_html('<div style="height:1px;background:#2a3040;margin:1.5rem 0"></div>')


def render_rule(margin: str = "1.2rem") -> None:
    render_html(f'<hr style="border-color:#1e2433;margin:{margin} 0">')


def render_openrouter_warning(settings: Settings) -> None:
    if settings.openrouter_api_key:
        return
    render_html(
        '<div style="background:rgba(239,83,80,0.09);border-left:4px solid #ef5350;'
        'padding:10px 16px;border-radius:8px;margin-top:8px;font-size:0.85rem;">'
        "⚠️ <b>OPENROUTER_API_KEY</b> не найден. Добавьте ключ в <code>.env</code> "
        "перед запуском чата.</div>"
    )


def metric_card_html(
    label: str,
    value: object,
    *,
    color: str = ACCENT_COLOR,
    font_size: str = "1.6rem",
) -> str:
    return (
        '<div style="background:#1a1f2e;border:1px solid #2a3040;border-radius:10px;'
        'padding:14px 16px;text-align:center;">'
        f'<div style="font-size:0.85rem;color:#8b92a8;margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:{font_size};font-weight:700;color:{color};">{value}</div>'
        "</div>"
    )
