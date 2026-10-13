"""Sidebar chat list: new chat, search, select, rename, delete, clear."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from rag_project.infrastructure.chat_store import (
    ChatMeta,
    create_chat,
    delete_chat,
    list_chats,
    rename_chat,
    save_messages,
    save_sources,
)
from rag_project.ui.formatting import plural_ru
from rag_project.ui.state import activate_chat
from rag_project.ui.theme import render_html


def render_chat_list(chat_root: Path, chats: list[ChatMeta], active_chat: ChatMeta) -> None:
    render_html(
        '<div class="sidebar-section"><span>Чаты</span>'
        f'<span class="sidebar-count">{plural_ru(len(chats), "чат", "чата", "чатов")}</span></div>'
    )
    if st.button("Новый чат", icon=":material/add:", type="primary", width="stretch"):
        activate_chat(create_chat(chat_root).id)
        st.rerun()

    search_query = (
        st.text_input(
            "Поиск чатов",
            placeholder="Найти чат…",
            label_visibility="collapsed",
            icon=":material/search:",
        )
        .strip()
        .lower()
    )
    visible_chats = [
        chat for chat in chats if not search_query or search_query in chat.title.lower()
    ]
    if search_query and not visible_chats:
        st.caption("Чаты не найдены")

    with st.container(key="chat-list", gap=None):
        for chat in visible_chats:
            _render_chat_row(chat, active_chat)

    _render_active_chat_actions(chat_root, active_chat)

    chat_to_delete = next(
        (chat for chat in chats if chat.id == st.session_state.delete_chat_id), None
    )
    if chat_to_delete is not None:
        _confirm_delete_chat(chat_root, chat_to_delete, active_chat)


def _render_chat_row(chat: ChatMeta, active_chat: ChatMeta) -> None:
    is_active = chat.id == active_chat.id
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        if st.button(
            chat.title,
            key=f"select_chat_{chat.id}",
            width="stretch",
            type="primary" if is_active else "tertiary",
            disabled=is_active,
            help="Активный чат" if is_active else None,
        ):
            activate_chat(chat.id)
            st.rerun()
        if st.button(
            ":material/delete:",
            key=f"delete_chat_{chat.id}",
            type="tertiary",
            help=f"Удалить чат «{chat.title}»",
        ):
            st.session_state.delete_chat_id = chat.id
            st.rerun()


def _render_active_chat_actions(chat_root: Path, active_chat: ChatMeta) -> None:
    with st.container(horizontal=True, gap="small", key="active-chat-actions"):
        with st.popover("Переименовать", icon=":material/edit:", width="stretch"):
            with st.form("rename_active_chat", border=False):
                new_title = st.text_input("Название активного чата", value=active_chat.title)
                renamed = st.form_submit_button("Сохранить", type="primary", width="stretch")
            if renamed:
                rename_chat(chat_root, active_chat.id, new_title)
                st.rerun()
        if st.button(
            "Очистить",
            icon=":material/clear_all:",
            width="stretch",
            help="Удалить сообщения и источники активного чата",
        ):
            _confirm_clear_chat(active_chat)


def _cancel_delete() -> None:
    st.session_state.delete_chat_id = None


@st.dialog("Удалить чат?", on_dismiss=_cancel_delete)
def _confirm_delete_chat(chat_root: Path, chat: ChatMeta, active_chat: ChatMeta) -> None:
    render_html(
        f"<p>Чат <b>«{html.escape(chat.title)}»</b> будет удалён вместе с историей, "
        "документами и индексом. Это действие нельзя отменить.</p>"
    )
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Отмена", key=f"cancel_delete_chat_{chat.id}"):
            _cancel_delete()
            st.rerun()
        if st.button("Удалить чат", key=f"confirm_delete_chat_{chat.id}", type="primary"):
            delete_chat(chat_root, chat.id)
            remaining = list_chats(chat_root) or [create_chat(chat_root)]
            remaining_by_id = {remaining_chat.id: remaining_chat for remaining_chat in remaining}
            next_chat = remaining_by_id.get(active_chat.id, remaining[0])
            st.session_state.delete_chat_id = None
            activate_chat(next_chat.id)
            st.rerun()


@st.dialog("Очистить чат?")
def _confirm_clear_chat(active_chat: ChatMeta) -> None:
    render_html(
        f"<p>Все сообщения и найденные источники чата <b>«{html.escape(active_chat.title)}»</b> "
        "будут удалены. Документы и индекс останутся.</p>"
    )
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Отмена", key="cancel_clear_chat"):
            st.rerun()
        if st.button("Очистить чат", key="confirm_clear_chat", type="primary"):
            st.session_state.messages = []
            st.session_state.last_context = []
            save_messages(active_chat.path, [])
            save_sources(active_chat.path, [])
            st.rerun()
