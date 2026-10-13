"""Settings page: answer parameters, reranker status, UI reset."""

from __future__ import annotations

import html

import streamlit as st

from rag_project.config import Settings
from rag_project.infrastructure.chat_store import ChatMeta
from rag_project.ui.state import invalidate_chat_session, reset_ui_state
from rag_project.ui.theme import render_html, render_openrouter_warning, render_rule


def render_settings_page(base_settings: Settings, active_chat: ChatMeta) -> None:
    st.markdown("### Настройки")
    st.caption(f"Активный чат: {active_chat.title}")

    col_model, col_key = st.columns(2)
    col_model.metric("LLM model", base_settings.openrouter_model)
    col_key.metric(
        "OpenRouter API key",
        "Найден" if base_settings.openrouter_api_key else "Не найден",
    )
    render_openrouter_warning(base_settings)

    st.markdown("#### Параметры ответа")
    st.session_state.top_k = st.number_input(
        "Top-K chunks",
        min_value=1,
        max_value=20,
        value=int(st.session_state.top_k),
        step=1,
    )
    st.session_state.max_context_chars = st.number_input(
        "MAX_CONTEXT_CHARS",
        min_value=1_000,
        max_value=50_000,
        value=int(st.session_state.max_context_chars),
        step=1_000,
    )
    st.session_state.temperature = st.slider(
        "TEMPERATURE",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.temperature),
        step=0.05,
    )

    if st.button("Применить настройки", type="primary"):
        invalidate_chat_session()
        st.success("Настройки применены")

    st.markdown("#### Reranker")
    col_enabled, col_candidates = st.columns(2)
    col_enabled.metric("Reranker", "ON" if base_settings.use_reranker else "OFF")
    col_candidates.metric("Rerank candidates", str(base_settings.rerank_candidates))
    if base_settings.use_reranker:
        render_html(
            '<div style="font-size:0.82rem;color:#8b92a8;word-break:break-all">'
            f"{html.escape(base_settings.reranker_model)}</div>"
        )

    render_rule()
    if st.button("Сбросить состояние UI"):
        reset_ui_state(active_chat.id)
        st.rerun()
