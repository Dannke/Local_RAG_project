"""Chat page: chat list (left), conversation (center), sources (right)."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from rag_project.application.chat import ChatSession
from rag_project.bootstrap import create_chat_session
from rag_project.config import Settings
from rag_project.domain.errors import (
    InvalidModelError,
    LLMClientError,
    LLMTimeoutError,
    MissingAPIKeyError,
)
from rag_project.infrastructure.chat_store import (
    ChatMeta,
    create_chat,
    delete_chat,
    list_chats,
    rename_chat,
    save_messages,
    save_search_results,
    save_sources,
)
from rag_project.infrastructure.rate_limit import RateLimitExceededError
from rag_project.ui.sources_panel import render_sources_panel
from rag_project.ui.state import activate_chat
from rag_project.ui.theme import ASSISTANT_AVATAR, USER_AVATAR, chat_avatar, render_html
from rag_project.ui.workspace import ChatWorkspace

logger = logging.getLogger(__name__)

# Most specific first: the first matching type decides the error title.
_LLM_ERROR_TITLES: tuple[tuple[type[LLMClientError], str], ...] = (
    (MissingAPIKeyError, "Ошибка конфигурации"),
    (RateLimitExceededError, "Превышен лимит запросов"),
    (LLMTimeoutError, "OpenRouter timeout"),
    (InvalidModelError, "Модель недоступна"),
    (LLMClientError, "Ошибка LLM"),
)


def render_chat_page(
    settings: Settings,
    workspace: ChatWorkspace,
    chat_root: Path,
    chats: list[ChatMeta],
) -> None:
    # Column weights: left (chat list) 15% · center (chat) 60% · right (sources) 25%
    left_panel, center_panel, right_panel = st.columns([1, 4, 1.67], gap="medium")

    with left_panel:
        _render_chat_list(chat_root, chats, workspace.chat)

    with center_panel:
        _render_conversation(settings, workspace)

    with right_panel:
        render_sources_panel(st.session_state.last_context)


# ─────────────────────────── Left panel ───────────────────────────


def _render_chat_list(chat_root: Path, chats: list[ChatMeta], active_chat: ChatMeta) -> None:
    """Chat management: new chat, search, select, rename, delete, clear."""
    render_html('<h2 style="font-size:1.1rem;margin-bottom:0.75rem;">Чаты</h2>')
    if st.button("+ Новый чат", use_container_width=True, type="primary"):
        activate_chat(create_chat(chat_root).id)
        st.rerun()

    search_query = (
        st.text_input("Поиск чатов", placeholder="Найти чат...", label_visibility="collapsed")
        .strip()
        .lower()
    )
    visible_chats = [
        chat for chat in chats if not search_query or search_query in chat.title.lower()
    ]
    if search_query and not visible_chats:
        st.caption("Чаты не найдены")

    for chat in visible_chats:
        _render_chat_row(chat_root, chat, active_chat)

    render_html("<br>")
    with st.form("rename_active_chat"):
        new_title = st.text_input("Название активного чата", value=active_chat.title)
        renamed = st.form_submit_button("Переименовать", use_container_width=True)
    if renamed:
        rename_chat(chat_root, active_chat.id, new_title)
        st.rerun()

    if st.button("Очистить чат", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_context = []
        save_messages(active_chat.path, [])
        save_sources(active_chat.path, [])
        st.rerun()


def _render_chat_row(chat_root: Path, chat: ChatMeta, active_chat: ChatMeta) -> None:
    is_active = chat.id == active_chat.id
    title = f"● {chat.title}" if is_active else chat.title
    col_chat, col_delete = st.columns([5, 1])
    if col_chat.button(
        title,
        key=f"select_chat_{chat.id}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
        disabled=is_active,
    ):
        activate_chat(chat.id)
        st.rerun()

    if col_delete.button("×", key=f"delete_chat_{chat.id}", help="Удалить чат"):
        st.session_state.delete_chat_id = chat.id
        st.rerun()

    if st.session_state.delete_chat_id != chat.id:
        return

    st.warning(f"Удалить чат «{chat.title}»?")
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("Да", key=f"confirm_delete_chat_{chat.id}"):
        delete_chat(chat_root, chat.id)
        remaining = list_chats(chat_root) or [create_chat(chat_root)]
        remaining_by_id = {remaining_chat.id: remaining_chat for remaining_chat in remaining}
        next_chat = remaining_by_id.get(active_chat.id, remaining[0])
        st.session_state.delete_chat_id = None
        activate_chat(next_chat.id)
        st.rerun()
    if cancel_col.button("Нет", key=f"cancel_delete_chat_{chat.id}"):
        st.session_state.delete_chat_id = None
        st.rerun()


# ─────────────────────────── Center panel ─────────────────────────


def _render_conversation(settings: Settings, workspace: ChatWorkspace) -> None:
    """Message history, question form and the streamed answer."""
    st.markdown("### Чат")

    if not workspace.index.exists():
        st.info("ℹ️ Сначала загрузите документы и создайте индекс.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=chat_avatar(message["role"])):
            st.markdown(message["content"])

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Вопрос",
            placeholder="Задайте вопрос о документе…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("➤ Отправить", type="primary")

    if not submitted:
        return
    if not question or not question.strip():
        st.warning("⚠️ Введите вопрос перед отправкой.")
        return

    question = question.strip()
    chat_path = workspace.chat.path
    _append_message(chat_path, "user", question)
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    try:
        session = _get_chat_session(settings, workspace.index_dir)
        with st.spinner("Ищу контекст, ранжирую источники и запрашиваю LLM..."):
            response = session.stream(question, top_k=int(st.session_state.top_k))
    except LLMClientError as exc:
        _show_llm_error(exc, chat_path)
        return
    except FileNotFoundError:
        st.error("Индекс не найден. Сначала выполните индексацию.")
        return
    except Exception as exc:
        logger.exception("chat_question_failed")
        st.error(f"Ошибка обработки вопроса: {exc}")
        return

    st.session_state.last_context = response.results
    save_search_results(chat_path, response.results)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        answer_box = st.empty()
        answer_box.markdown("печатает...")
        answer = ""
        try:
            for token in response.chunks:
                answer += token
                answer_box.markdown(answer + "▌")
            answer_box.markdown(answer)
        except LLMClientError as exc:
            _show_llm_error(exc, chat_path)
            return

    _append_message(chat_path, "assistant", answer)


def _append_message(chat_path: Path, role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})
    save_messages(chat_path, st.session_state.messages)


def _get_chat_session(settings: Settings, index_dir: Path) -> ChatSession:
    """Return the cached chat session, rebuilding it when its inputs change."""
    signature = _chat_session_signature(settings, index_dir)
    if (
        st.session_state.chat_session is None
        or st.session_state.chat_session_signature != signature
    ):
        try:
            st.session_state.chat_session = create_chat_session(
                index_dir=index_dir,
                settings=settings,
            )
            st.session_state.chat_session_signature = signature
        except FileNotFoundError:
            raise
        except RuntimeError as exc:
            # Likely a missing dependency or a corrupt index.
            raise RuntimeError(f"Ошибка инициализации компонентов: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Не удалось создать сессию чата: {exc}") from exc
    return st.session_state.chat_session


def _chat_session_signature(settings: Settings, index_dir: Path) -> tuple:
    return (
        str(index_dir.resolve()),
        settings.openrouter_model,
        settings.max_context_chars,
        settings.temperature,
        settings.openrouter_timeout_seconds,
        settings.use_reranker,
        settings.reranker_model,
        settings.rerank_candidates,
    )


def _show_llm_error(exc: LLMClientError, chat_path: Path) -> None:
    title = next(title for error_type, title in _LLM_ERROR_TITLES if isinstance(exc, error_type))
    st.error(f"❌ {title}: {exc}")
    _append_message(chat_path, "assistant", f"{title}: {exc}")
