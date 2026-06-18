"""Streamlit UI for the local RAG project — enhanced visual edition."""

from __future__ import annotations

import sys
import warnings

# Suppress optional dependency warnings from transformers BEFORE any imports
warnings.filterwarnings('ignore', category=ImportWarning)
warnings.filterwarnings('ignore', message='.*torchvision.*')
warnings.filterwarnings('ignore', module='.*transformers.*')

from dataclasses import replace
from pathlib import Path

import streamlit as st

# Add src directory to Python path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_project.chat_store import (
    ChatMeta,
    chats_root,
    create_chat,
    delete_chat,
    ensure_chat_layout,
    get_chat_documents_dir,
    get_chat_index_dir,
    initialize_chats,
    list_chats,
    load_messages,
    rename_chat,
    save_messages,
)
from rag_project.config import Settings, load_settings
from rag_project.llm.llm_client import (
    InvalidModelError,
    LLMClientError,
    LLMTimeoutError,
    MissingAPIKeyError,
)
from rag_project.pipelines.chat_pipeline import ChatSession
from rag_project.pipelines.ingest_pipeline import ingest_to_faiss
from rag_project.retrieval.citations import build_citations

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
BASE_DIR = Path(__file__).parent
STYLE_PATH = BASE_DIR / "assets" / "styles.css"

_USER_AVATAR = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "viewBox='0 0 24 24' fill='none' stroke='%23d9d9d9' stroke-width='1.7' "
    "stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 "
    "4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/%3E%3Ccircle cx='12' cy='7' r='4'/%3E%3C/svg%3E"
)
_ASSISTANT_AVATAR = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "viewBox='0 0 24 24' fill='none' stroke='%23d9d9d9' stroke-width='1.7' "
    "stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='11' "
    "width='18' height='10' rx='2'/%3E%3Ccircle cx='12' cy='5' r='2'/%3E%3Cpath "
    "d='M12 7v4'/%3E%3Cpath d='M8 16h.01'/%3E%3Cpath d='M16 16h.01'/%3E%3C/svg%3E"
)


def _inject_css() -> None:
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _chat_avatar(role: str) -> str:
    if role == "user":
        return _USER_AVATAR
    return _ASSISTANT_AVATAR


# ─────────────────────────── Main ─────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Local RAG",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    _init_state()

    base_settings = load_settings()

    # Проверка атрибутов для диагностики
    has_use_reranker = hasattr(base_settings, 'use_reranker')
    has_reranker_model = hasattr(base_settings, 'reranker_model')
    has_rerank_candidates = hasattr(base_settings, 'rerank_candidates')

    if not (has_use_reranker and has_reranker_model and has_rerank_candidates):
        import sys as _sys
        print("WARNING: Settings объект имеет неполные атрибуты!", file=_sys.stderr)
        print(f"  - use_reranker: {has_use_reranker}", file=_sys.stderr)
        print(f"  - reranker_model: {has_reranker_model}", file=_sys.stderr)
        print(f"  - rerank_candidates: {has_rerank_candidates}", file=_sys.stderr)

    chat_root = chats_root(base_settings.project_root)
    chats = initialize_chats(chat_root)
    active_chat = _sync_active_chat(chat_root, chats)
    settings = _settings_from_state(base_settings)
    raw_data_dir = get_chat_documents_dir(active_chat.path)
    index_dir = get_chat_index_dir(active_chat.path)

    def chat_page() -> None:
        _render_chat_page(settings, index_dir, chat_root, chats, active_chat)

    def documents_page() -> None:
        _render_documents_page(settings, raw_data_dir, index_dir, active_chat)

    def settings_page() -> None:
        _render_settings_page(base_settings, active_chat)

    page = st.navigation(
        [
            st.Page(chat_page, title="Chat", icon="💬", url_path="chat", default=True),
            st.Page(documents_page, title="Documents", icon="📄", url_path="documents"),
            st.Page(settings_page, title="Settings", icon="⚙️", url_path="settings"),
        ],
        position="sidebar",
        expanded=True,
    )
    _render_sidebar(active_chat)
    _render_app_header()
    page.run()


def _render_app_header() -> None:
    st.markdown(
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
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    settings = load_settings()
    defaults = {
        "messages": [],
        "last_context": [],
        "uploaded_files": [],
        "index_ready": False,
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


def _reset_ui_state() -> None:
    st.session_state.messages = []
    st.session_state.last_context = []
    st.session_state.uploaded_files = []
    st.session_state.chat_session = None
    st.session_state.chat_session_signature = None
    st.session_state.notice = None


def _sync_active_chat(chat_root: Path, chats: list[ChatMeta]) -> ChatMeta:
    chat_by_id = {chat.id: chat for chat in chats}
    active_chat_id = st.session_state.active_chat_id
    if active_chat_id not in chat_by_id:
        active_chat_id = chats[0].id
        st.session_state.active_chat_id = active_chat_id

    active_chat = chat_by_id[active_chat_id]
    ensure_chat_layout(active_chat.path)

    if st.session_state.loaded_chat_id != active_chat.id:
        st.session_state.loaded_chat_id = active_chat.id
        st.session_state.messages = load_messages(active_chat.path)
        st.session_state.uploaded_files = [
            path.name for path in _list_documents(get_chat_documents_dir(active_chat.path))
        ]
        _reset_index_dependent_state(clear_messages=False)

    return active_chat


def _settings_from_state(settings: Settings) -> Settings:
    return replace(
        settings,
        top_k=int(st.session_state.top_k),
        max_context_chars=int(st.session_state.max_context_chars),
        temperature=float(st.session_state.temperature),
    )


# ─────────────────────────── Pages ────────────────────────────────

def _render_chat_page(
    settings: Settings,
    index_dir: Path,
    chat_root: Path,
    chats: list[ChatMeta],
    active_chat: ChatMeta,
) -> None:
    left_panel, center_panel, right_panel = st.columns([1, 4.67, 1], gap="medium")

    with left_panel:
        _render_left_panel(chat_root, chats, active_chat)

    with center_panel:
        _render_center_panel(settings, index_dir, active_chat)

    with right_panel:
        _render_right_panel(st.session_state.last_context)


def _render_documents_page(
    settings: Settings,
    raw_data_dir: Path,
    index_dir: Path,
    active_chat: ChatMeta,
) -> None:
    st.markdown("### Документы")
    _render_index_status(raw_data_dir, index_dir, active_chat)
    st.markdown(
        '<hr style="border-color:#1e2433;margin:1.2rem 0">',
        unsafe_allow_html=True,
    )
    _render_upload_and_index(settings, raw_data_dir, index_dir, active_chat)


def _render_settings_page(base_settings: Settings, active_chat: ChatMeta) -> None:
    st.markdown("### Настройки")
    st.caption(f"Активный чат: {active_chat.title}")

    col_model, col_key = st.columns(2)
    col_model.metric("LLM model", base_settings.openrouter_model)
    col_key.metric(
        "OpenRouter API key",
        "Найден" if base_settings.openrouter_api_key else "Не найден",
    )
    _render_openrouter_warning(base_settings)

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
        st.session_state.chat_session = None
        st.session_state.chat_session_signature = None
        st.success("Настройки применены")

    st.markdown("#### Reranker")
    use_reranker = getattr(base_settings, 'use_reranker', True)
    reranker_model = getattr(
        base_settings,
        'reranker_model',
        'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
    )
    rerank_candidates = getattr(base_settings, 'rerank_candidates', 20)

    col_enabled, col_candidates = st.columns(2)
    col_enabled.metric("Reranker", "ON" if use_reranker else "OFF")
    col_candidates.metric("Rerank candidates", str(rerank_candidates))
    if use_reranker:
        st.markdown(
            f'<div style="font-size:0.82rem;color:#8b92a8;word-break:break-all">'
            f'{reranker_model}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<hr style="border-color:#1e2433;margin:1.2rem 0">',
        unsafe_allow_html=True,
    )
    if st.button("Сбросить состояние UI"):
        _reset_ui_state()
        st.session_state.active_chat_id = active_chat.id
        st.session_state.loaded_chat_id = None
        st.session_state.uploader_versions = {}
        st.rerun()


# ─────────────────────────── Sidebar ──────────────────────────────

def _render_sidebar(active_chat: ChatMeta) -> None:
    with st.sidebar:
        st.markdown('<hr style="border-color:#1e2433;margin:1rem 0">', unsafe_allow_html=True)
        st.caption(f"Активный чат: {active_chat.title}")


def _render_left_panel(
    chat_root: Path,
    chats: list[ChatMeta],
    active_chat: ChatMeta,
) -> None:
    """LEFT PANEL: Chat management (new chat, chat list, search, rename, delete)."""
    st.markdown(
        '<h2 style="font-size:1.1rem;margin-bottom:0.75rem;">Чаты</h2>',
        unsafe_allow_html=True,
    )
    if st.button("+ Новый чат", use_container_width=True, type="primary"):
        chat = create_chat(chat_root)
        _activate_chat(chat.id)
        st.rerun()

    search_query = st.text_input(
        "Поиск чатов",
        placeholder="Найти чат...",
        label_visibility="collapsed",
    ).strip().lower()
    visible_chats = [
        chat for chat in chats if not search_query or search_query in chat.title.lower()
    ]

    if search_query and not visible_chats:
        st.caption("Чаты не найдены")

    for chat in visible_chats:
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
            _activate_chat(chat.id)
            st.rerun()

        if col_delete.button("×", key=f"delete_chat_{chat.id}", help="Удалить чат"):
            st.session_state.delete_chat_id = chat.id
            st.rerun()

        if st.session_state.delete_chat_id == chat.id:
            st.warning(f"Удалить чат «{chat.title}»?")
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("Да", key=f"confirm_delete_chat_{chat.id}"):
                delete_chat(chat_root, chat.id)
                remaining = list_chats(chat_root)
                if not remaining:
                    remaining = [create_chat(chat_root)]
                remaining_by_id = {remaining_chat.id: remaining_chat for remaining_chat in remaining}
                next_chat = remaining_by_id.get(active_chat.id, remaining[0])
                st.session_state.delete_chat_id = None
                _activate_chat(next_chat.id)
                st.rerun()
            if cancel_col.button("Нет", key=f"cancel_delete_chat_{chat.id}"):
                st.session_state.delete_chat_id = None
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
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
        st.rerun()


def _activate_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id
    st.session_state.loaded_chat_id = None
    st.session_state.delete_chat_id = None
    st.session_state.last_context = []
    st.session_state.uploaded_files = []
    st.session_state.chat_session = None
    st.session_state.chat_session_signature = None


def _uploader_key(chat_id: str) -> str:
    versions = st.session_state.setdefault("uploader_versions", {})
    version = int(versions.get(chat_id, 0))
    return f"document_uploader_{chat_id}_{version}"


def _bump_uploader_version(chat_id: str) -> None:
    versions = st.session_state.setdefault("uploader_versions", {})
    versions[chat_id] = int(versions.get(chat_id, 0)) + 1


# ─────────────────────────── Status ───────────────────────────────

def _render_status(
    settings: Settings,
    raw_data_dir: Path,
    index_dir: Path,
    active_chat: ChatMeta,
) -> None:
    _render_index_status(raw_data_dir, index_dir, active_chat)
    _render_openrouter_warning(settings)


def _render_index_status(
    raw_data_dir: Path,
    index_dir: Path,
    active_chat: ChatMeta,
) -> None:
    documents = _list_documents(raw_data_dir)
    index_ready = _index_exists(index_dir)
    st.session_state.index_ready = index_ready

    st.markdown("### Статус")
    col1, col2 = st.columns(2)
    col1.metric("Индекс", "Готов ✓" if index_ready else "Не создан")
    col2.metric("Документы", str(len(documents)))
    st.caption(f"Активный чат: {active_chat.title}")


def _render_openrouter_warning(settings: Settings) -> None:
    if not settings.openrouter_api_key:
        st.markdown(
            '<div style="background:rgba(239,83,80,0.09);border-left:4px solid #ef5350;'
            'padding:10px 16px;border-radius:8px;margin-top:8px;font-size:0.85rem;">'
            '⚠️ <b>OPENROUTER_API_KEY</b> не найден. Добавьте ключ в <code>.env</code> '
            'перед запуском чата.</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────── Upload & Index ───────────────────────

def _render_upload_and_index(
    settings: Settings,
    raw_data_dir: Path,
    index_dir: Path,
    active_chat: ChatMeta,
) -> None:
    if st.session_state.notice:
        st.success(st.session_state.notice)
        st.session_state.notice = None

    uploaded_files = st.file_uploader(
        "Загрузите PDF, DOCX, TXT или MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=_uploader_key(active_chat.id),
    )

    if uploaded_files:
        _save_uploaded_files(uploaded_files, raw_data_dir)

    if st.session_state.uploaded_files:
        files_list = ", ".join(st.session_state.uploaded_files)
        st.markdown(
            f'<div style="font-size:0.82rem;color:#8b92a8;margin:0.4rem 0">'
            f'Файлы в этой сессии: {files_list}</div>',
            unsafe_allow_html=True,
        )

    _render_document_list(raw_data_dir, index_dir, active_chat)

    clear_index = st.checkbox("Также удалить FAISS-индекс", value=True)
    if st.button("🗑 Очистить все загруженные документы"):
        try:
            removed_documents = _clear_uploaded_documents(raw_data_dir)
            removed_index_files = _clear_index_files(index_dir) if clear_index else 0
            _reset_index_dependent_state()
            st.session_state.uploaded_files = []
            _bump_uploader_version(active_chat.id)
            st.session_state.index_ready = _index_exists(index_dir)
            st.session_state.notice = (
                f"Удалено документов: {removed_documents}. "
                f"Файлов индекса: {removed_index_files}."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Ошибка очистки: {exc}")

    if st.button("Индексировать документы", type="primary"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(stage: str, current: int, total: int) -> None:
                if total > 0:
                    progress_bar.progress(current / total, text=f"{stage}: {current}/{total}")
                else:
                    status_text.text(stage)

            status_text.text("Инициализация индексирования...")
            count = ingest_to_faiss(
                input_dir=raw_data_dir,
                index_dir=index_dir,
                settings=settings,
                incremental=True,
                progress_callback=update_progress,
            )
            progress_bar.progress(1.0, text="Завершено!")
            st.session_state.index_ready = True
            st.session_state.chat_session = None
            st.session_state.chat_session_signature = None
            st.success(f"Индекс обновлён. Chunks: {count}")
        except Exception as exc:
            st.session_state.index_ready = False
            st.error(f"Ошибка индексации: {exc}")

    st.markdown(
        '<hr style="border-color:#1e2433;margin:1.2rem 0">',
        unsafe_allow_html=True,
    )


def _save_uploaded_files(uploaded_files, raw_data_dir: Path) -> None:
    saved: list[str] = []
    errors: list[str] = []
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            errors.append(f"{filename}: неподдерживаемый формат")
            continue
        target = raw_data_dir / filename
        target.write_bytes(uploaded_file.getbuffer())
        saved.append(filename)

    if saved:
        st.session_state.uploaded_files = sorted(set(st.session_state.uploaded_files + saved))
        st.success("Загружено: " + ", ".join(saved))
    for error in errors:
        st.error(error)


def _render_document_list(raw_data_dir: Path, index_dir: Path, active_chat: ChatMeta) -> None:
    documents = _list_documents(raw_data_dir)
    if not documents:
        st.markdown(
            '<div style="color:#5a6078;font-size:0.85rem;padding:0.5rem 0">'
            'Загруженных документов пока нет.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown("**Загруженные документы**")
    for document_path in documents:
        relative_path = document_path.relative_to(raw_data_dir).as_posix()
        size_kb = document_path.stat().st_size / 1024
        col_name, col_size, col_action = st.columns([5, 1, 1])
        col_name.write(relative_path)
        col_size.write(f"{size_kb:.1f} KB")
        if col_action.button("Удалить", key=f"delete_doc_{relative_path}"):
            try:
                _delete_single_document(document_path, raw_data_dir, index_dir)
                _bump_uploader_version(active_chat.id)
                st.session_state.notice = (
                    f"Документ удалён: {relative_path}. "
                    "Индекс сброшен, выполните индексацию заново."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Ошибка удаления файла: {exc}")


def _delete_single_document(path: Path, raw_data_dir: Path, index_dir: Path) -> None:
    _safe_unlink(path, raw_data_dir)
    _clear_index_files(index_dir)
    st.session_state.uploaded_files = [
        name for name in st.session_state.uploaded_files if name != path.name
    ]
    _reset_index_dependent_state()
    st.session_state.index_ready = False


# ─────────────────────────── Chat ─────────────────────────────────

def _render_center_panel(settings: Settings, index_dir: Path, active_chat: ChatMeta) -> None:
    """CENTER PANEL: Conversation (message history, streaming responses, chat input)."""
    st.markdown("### Чат")

    if not _index_exists(index_dir):
        st.info("ℹ️ Сначала загрузите документы и создайте индекс.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=_chat_avatar(message["role"])):
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
    st.session_state.messages.append({"role": "user", "content": question})
    save_messages(active_chat.path, st.session_state.messages)
    with st.chat_message("user", avatar=_USER_AVATAR):
        st.markdown(question)

    try:
        session = _get_chat_session(settings, index_dir)
        with st.spinner("Ищу контекст, ранжирую источники и запрашиваю LLM..."):
            response = session.stream(question, top_k=int(st.session_state.top_k))
    except MissingAPIKeyError as exc:
        _show_llm_error("Ошибка конфигурации", exc, active_chat.path)
        return
    except LLMTimeoutError as exc:
        _show_llm_error("OpenRouter timeout", exc, active_chat.path)
        return
    except InvalidModelError as exc:
        _show_llm_error("Модель недоступна", exc, active_chat.path)
        return
    except LLMClientError as exc:
        _show_llm_error("Ошибка LLM", exc, active_chat.path)
        return
    except FileNotFoundError:
        st.error("Индекс не найден. Сначала выполните индексацию.")
        return
    except Exception as exc:
        st.error(f"Ошибка обработки вопроса: {exc}")
        return

    st.session_state.last_context = response.results

    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        answer_box = st.empty()
        answer_box.markdown("печатает...")
        answer = ""
        try:
            for token in response.chunks:
                answer += token
                answer_box.markdown(answer + "▌")
            answer_box.markdown(answer)
        except MissingAPIKeyError as exc:
            _show_llm_error("Ошибка конфигурации", exc, active_chat.path)
            return
        except LLMTimeoutError as exc:
            _show_llm_error("OpenRouter timeout", exc, active_chat.path)
            return
        except InvalidModelError as exc:
            _show_llm_error("Модель недоступна", exc, active_chat.path)
            return
        except LLMClientError as exc:
            _show_llm_error("Ошибка LLM", exc, active_chat.path)
            return

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_messages(active_chat.path, st.session_state.messages)


def _get_chat_session(settings: Settings, index_dir: Path) -> ChatSession:
    signature = _chat_session_signature(settings, index_dir)
    if (
        st.session_state.chat_session is None
        or st.session_state.chat_session_signature != signature
    ):
        st.session_state.chat_session = ChatSession.from_faiss_index(
            index_dir=index_dir,
            settings=settings,
        )
        st.session_state.chat_session_signature = signature
    return st.session_state.chat_session


def _chat_session_signature(settings: Settings, index_dir: Path) -> tuple:
    return (
        str(index_dir.resolve()),
        settings.openrouter_model,
        settings.max_context_chars,
        settings.temperature,
        settings.openrouter_timeout_seconds,
        getattr(settings, 'use_reranker', True),
        getattr(settings, 'reranker_model', 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'),
        getattr(settings, 'rerank_candidates', 20),
    )


def _show_llm_error(title: str, exc: Exception, chat_dir: Path) -> None:
    st.error(f"❌ {title}: {exc}")
    st.session_state.messages.append({"role": "assistant", "content": f"{title}: {exc}"})
    save_messages(chat_dir, st.session_state.messages)


# ─────────────────────────── Sources ──────────────────────────────

def _render_right_panel(results) -> None:
    """RIGHT PANEL: Citations panel (retrieved chunks, source metadata, PDF page numbers, relevance scores)."""
    st.markdown("### Источники")
    if not results:
        st.markdown(
            '<div style="color:#5a6078;font-size:0.85rem">'
            'Источники не найдены.</div>',
            unsafe_allow_html=True,
        )
        return

    for index, citation in enumerate(build_citations(results), start=1):
        page = citation.page or "n/a"
        score_bar = min(int(citation.score * 100), 100)
        if score_bar > 70:
            score_color = "#66bb6a"
        elif score_bar > 40:
            score_color = "#ffa726"
        else:
            score_color = "#ef5350"

        st.markdown(
            f'<div style="background:#1a1f2e;border:1px solid #2a3040;'
            f'border-radius:10px;padding:14px 18px;margin-bottom:10px;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="background:rgba(124,77,255,0.13);color:#7C4DFF;'
            f'font-size:0.75rem;font-weight:600;padding:2px 8px;border-radius:6px;">'
            f'{citation.label}</span>'
            f'<span style="font-weight:500;font-size:0.9rem;">{citation.source}</span>'
            f'</div>'
            f'<div style="display:flex;gap:16px;font-size:0.78rem;color:#8b92a8;'
            f'margin-bottom:8px;">'
            f'<span>Page: {page}</span>'
            f'<span>Chunk: {citation.chunk_index}</span>'
            f'<span>Relevance: <b style="color:{score_color}">'
            f'{citation.score:.4f}</b></span>'
            f'</div>'
            f'<div style="background:#0e1117;border-radius:6px;padding:8px 12px;'
            f'font-size:0.82rem;color:#b0b8c8;word-break:break-word;">'
            f'{citation.text[:300]}{"…" if len(citation.text) > 300 else ""}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Показать полный фрагмент"):
            st.write(citation.text)


# ─────────────────────────── Utilities ────────────────────────────

def _list_documents(raw_data_dir: Path) -> list[Path]:
    if not raw_data_dir.exists():
        return []
    return [
        path
        for path in raw_data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _clear_uploaded_documents(raw_data_dir: Path) -> int:
    removed = 0
    for path in _list_documents(raw_data_dir):
        _safe_unlink(path, raw_data_dir)
        removed += 1
    return removed


def _clear_index_files(index_dir: Path) -> int:
    if not index_dir.exists():
        return 0
    removed = 0
    for path in (index_dir / "index.faiss", index_dir / "documents.json", index_dir / "manifest.json"):
        if path.exists():
            _safe_unlink(path, index_dir)
            removed += 1
    return removed


def _reset_index_dependent_state(clear_messages: bool = False) -> None:
    if clear_messages:
        st.session_state.messages = []
    st.session_state.last_context = []
    st.session_state.chat_session = None
    st.session_state.chat_session_signature = None


def _safe_unlink(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise RuntimeError(f"Refusing to delete outside {resolved_root}: {resolved_path}")
    if resolved_path.is_file():
        resolved_path.unlink()


def _index_exists(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "documents.json").exists()


if __name__ == "__main__":
    main()
