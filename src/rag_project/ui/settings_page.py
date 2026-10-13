"""Settings page: answer parameters, reranker status, UI reset."""

from __future__ import annotations

import html

import streamlit as st

from rag_project.config import Settings
from rag_project.infrastructure.chat_store import ChatMeta
from rag_project.ui.state import invalidate_chat_session, reset_ui_state
from rag_project.ui.theme import (
    kpi_card_html,
    render_html,
    render_openrouter_warning,
    render_page_header,
    render_section_title,
)


def render_settings_page(base_settings: Settings, active_chat: ChatMeta) -> None:
    render_page_header("Настройки", f"Активный чат: «{html.escape(active_chat.title)}»")

    render_html(
        '<div class="kpi-grid kpi-grid-2">'
        + kpi_card_html("LLM model", html.escape(base_settings.openrouter_model))
        + kpi_card_html(
            "OpenRouter API key",
            "Найден" if base_settings.openrouter_api_key else "Не найден",
            tone="ok" if base_settings.openrouter_api_key else "warn",
        )
        + "</div>"
    )
    render_openrouter_warning(base_settings)

    render_section_title("Параметры ответа", "применяются к следующему вопросу")
    with st.container(border=True, key="answer-params"):
        st.session_state.top_k = st.number_input(
            "Top-K chunks",
            min_value=1,
            max_value=20,
            value=int(st.session_state.top_k),
            step=1,
            help="Сколько фрагментов документов передавать модели",
        )
        st.session_state.max_context_chars = st.number_input(
            "MAX_CONTEXT_CHARS",
            min_value=1_000,
            max_value=50_000,
            value=int(st.session_state.max_context_chars),
            step=1_000,
            help="Максимальный объём контекста в символах",
        )
        st.session_state.temperature = st.slider(
            "TEMPERATURE",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.temperature),
            step=0.05,
            help="Ниже — точнее и суше, выше — свободнее формулировки",
        )

        if st.button("Применить настройки", type="primary", icon=":material/check:"):
            invalidate_chat_session()
            st.success("Настройки применены")

    render_section_title("Reranker")
    reranker_note = html.escape(base_settings.reranker_model) if base_settings.use_reranker else ""
    render_html(
        '<div class="kpi-grid kpi-grid-2">'
        + kpi_card_html(
            "Reranker",
            "ON" if base_settings.use_reranker else "OFF",
            tone="ok" if base_settings.use_reranker else "",
            note=reranker_note,
        )
        + kpi_card_html("Rerank candidates", base_settings.rerank_candidates)
        + "</div>"
    )

    render_section_title("Сброс")
    if st.button(
        "Сбросить состояние UI",
        icon=":material/restart_alt:",
        help="Очистить состояние интерфейса активного чата; файлы на диске не меняются",
    ):
        reset_ui_state(active_chat.id)
        st.rerun()
