"""Chat page: conversation (left) and the sources of the last answer (right)."""

from __future__ import annotations

import html
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
from rag_project.infrastructure.chat_store import save_messages, save_search_results
from rag_project.infrastructure.rate_limit import RateLimitExceededError
from rag_project.ui.formatting import plural_ru
from rag_project.ui.sources_panel import render_sources_panel
from rag_project.ui.theme import (
    ASSISTANT_AVATAR,
    TYPING_INDICATOR_HTML,
    USER_AVATAR,
    chat_avatar,
    empty_state_html,
    render_html,
    render_page_header,
)
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


def render_chat_page(settings: Settings, workspace: ChatWorkspace) -> None:
    with st.container(key="chat-layout"):
        center_panel, right_panel = st.columns([2.3, 1], gap="large")

        with center_panel:
            _render_conversation(settings, workspace)

        with right_panel, st.container(key="sources-panel"):
            render_sources_panel(st.session_state.last_context)


def _render_conversation(settings: Settings, workspace: ChatWorkspace) -> None:
    """Message history, the pinned question box and the streamed answer."""
    messages = st.session_state.messages
    documents = len(workspace.documents.list_files())
    render_page_header(
        html.escape(workspace.chat.title),
        plural_ru(documents, "документ", "документа", "документов") + " в базе знаний",
    )

    if not workspace.index.exists():
        render_html(
            empty_state_html(
                "Сначала нужен индекс",
                "Загрузите документы на странице «Документы» и нажмите «Индексировать». "
                "После этого здесь можно будет задавать вопросы.",
                mark="§",
            )
        )
        return

    thread = st.container(key="chat-thread", height=560)
    with thread:
        empty_slot = st.empty()
        if not messages:
            empty_slot.markdown(
                empty_state_html(
                    "Спросите о своих документах",
                    "Ответ строится только по документам этого чата. "
                    "Фрагменты, на которые он опирается, появятся справа как S1, S2…",
                    mark="?",
                ),
                unsafe_allow_html=True,
            )
        for message in messages:
            with st.chat_message(message["role"], avatar=chat_avatar(message["role"])):
                st.markdown(message["content"])

    question = st.chat_input("Задайте вопрос о документах…")
    if question is None:
        return
    question = question.strip()
    if not question:
        st.warning("Введите вопрос перед отправкой.")
        return

    chat_path = workspace.chat.path
    empty_slot.empty()
    with thread:
        _append_message(chat_path, "user", question)
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)
        _answer(settings, workspace, question)


def _answer(settings: Settings, workspace: ChatWorkspace, question: str) -> None:
    chat_path = workspace.chat.path
    pending = st.empty()
    with pending.container(), st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        answer_box = st.empty()
        answer_box.markdown(TYPING_INDICATOR_HTML, unsafe_allow_html=True)

    try:
        session = _get_chat_session(settings, workspace.index_dir)
        response = session.stream(question, top_k=int(st.session_state.top_k))
    except LLMClientError as exc:
        pending.empty()
        _show_llm_error(exc, chat_path)
        return
    except FileNotFoundError:
        pending.empty()
        st.error("Индекс не найден. Сначала выполните индексацию.")
        return
    except Exception as exc:
        pending.empty()
        logger.exception("chat_question_failed")
        st.error(f"Ошибка обработки вопроса: {exc}")
        return

    st.session_state.last_context = response.results
    save_search_results(chat_path, response.results)

    answer = ""
    text_box = None
    try:
        for token in response.chunks:
            answer += token
            if text_box is None:
                # The keyed wrapper draws the caret in CSS, so the answer
                # itself stays plain markdown without unsafe HTML.
                with answer_box.container(key="answer-streaming"):
                    text_box = st.empty()
            text_box.markdown(answer)
    except LLMClientError as exc:
        answer_box.markdown(answer)  # drop the caret from the partial answer
        _show_llm_error(exc, chat_path)
        return
    answer_box.markdown(answer)

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
