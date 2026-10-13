"""Streamlit application shell: page config, state bootstrap and navigation."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_project.config import load_settings
from rag_project.infrastructure.chat_store import chats_root, initialize_chats
from rag_project.logging_setup import setup_logging
from rag_project.ui.chat_list import render_chat_list
from rag_project.ui.chat_page import render_chat_page
from rag_project.ui.documents_page import render_documents_page
from rag_project.ui.settings_page import render_settings_page
from rag_project.ui.state import init_state, settings_from_state, sync_active_chat
from rag_project.ui.theme import inject_css


def main(style_path: Path) -> None:
    st.set_page_config(
        page_title="Local RAG",
        page_icon=":material/menu_book:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Settings first: loading them applies .env, which logging setup reads.
    base_settings = load_settings()
    setup_logging()
    inject_css(style_path)
    st.logo(style_path.with_name("logo.svg"), size="large")
    init_state(base_settings)

    chat_root = chats_root(base_settings.project_root)
    chats = initialize_chats(chat_root)
    workspace = sync_active_chat(chats)
    settings = settings_from_state(base_settings)

    def chat_page() -> None:
        render_chat_page(settings, workspace)

    def documents_page() -> None:
        render_documents_page(settings, workspace)

    def settings_page() -> None:
        render_settings_page(base_settings, workspace.chat)

    page = st.navigation(
        [
            st.Page(chat_page, title="Чат", icon=":material/forum:", url_path="chat", default=True),
            st.Page(
                documents_page,
                title="Документы",
                icon=":material/description:",
                url_path="documents",
            ),
            st.Page(settings_page, title="Настройки", icon=":material/tune:", url_path="settings"),
        ],
        position="sidebar",
        expanded=True,
    )
    with st.sidebar:
        render_chat_list(chat_root, chats, workspace.chat)
    page.run()
