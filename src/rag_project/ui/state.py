"""Streamlit session-state management."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from rag_project.config import Settings
from rag_project.infrastructure.chat_store import (
    ChatMeta,
    ensure_chat_layout,
    load_messages,
    load_search_results,
)
from rag_project.ui.workspace import ChatWorkspace


def init_state(settings: Settings) -> None:
    defaults = {
        "messages": [],
        "last_context": [],
        "uploaded_files": [],
        "chat_session": None,
        "chat_session_signature": None,
        "notice": None,
        "active_chat_id": None,
        "loaded_chat_id": None,
        "delete_chat_id": None,
        "uploader_versions": {},
        "top_k": settings.top_k,
        "max_context_chars": settings.max_context_chars,
        "temperature": settings.temperature,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def settings_from_state(settings: Settings) -> Settings:
    """Apply the answer parameters edited on the Settings page."""
    return replace(
        settings,
        top_k=int(st.session_state.top_k),
        max_context_chars=int(st.session_state.max_context_chars),
        temperature=float(st.session_state.temperature),
    )


def sync_active_chat(chats: list[ChatMeta]) -> ChatWorkspace:
    """Resolve the active chat and load its persisted state on first visit."""
    chat_by_id = {chat.id: chat for chat in chats}
    active_chat_id = st.session_state.active_chat_id
    if active_chat_id not in chat_by_id:
        active_chat_id = chats[0].id
        st.session_state.active_chat_id = active_chat_id

    workspace = ChatWorkspace.for_chat(chat_by_id[active_chat_id])
    chat_path = workspace.chat.path
    ensure_chat_layout(chat_path)

    if st.session_state.loaded_chat_id != workspace.chat.id:
        st.session_state.loaded_chat_id = workspace.chat.id
        st.session_state.messages = load_messages(chat_path)
        st.session_state.uploaded_files = [path.name for path in workspace.documents.list_files()]
        reset_index_dependent_state()
        # Reload persisted sources after the index-dependent reset so they
        # survive a page refresh; they only clear on chat delete/clear.
        st.session_state.last_context = load_search_results(chat_path)

    return workspace


def activate_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id
    st.session_state.loaded_chat_id = None
    st.session_state.delete_chat_id = None
    st.session_state.last_context = []
    st.session_state.uploaded_files = []
    invalidate_chat_session()


def invalidate_chat_session() -> None:
    """Drop the cached chat session so the next question rebuilds it."""
    st.session_state.chat_session = None
    st.session_state.chat_session_signature = None


def reset_index_dependent_state() -> None:
    st.session_state.last_context = []
    invalidate_chat_session()


def reset_ui_state(active_chat_id: str) -> None:
    st.session_state.messages = []
    st.session_state.last_context = []
    st.session_state.uploaded_files = []
    st.session_state.notice = None
    invalidate_chat_session()
    st.session_state.active_chat_id = active_chat_id
    st.session_state.loaded_chat_id = None
    st.session_state.uploader_versions = {}


def uploader_key(chat_id: str) -> str:
    versions = st.session_state.setdefault("uploader_versions", {})
    version = int(versions.get(chat_id, 0))
    return f"document_uploader_{chat_id}_{version}"


def bump_uploader_version(chat_id: str) -> None:
    """Reset the file uploader widget (a new key renders an empty uploader)."""
    versions = st.session_state.setdefault("uploader_versions", {})
    versions[chat_id] = int(versions.get(chat_id, 0)) + 1

