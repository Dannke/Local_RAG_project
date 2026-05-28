"""Streamlit UI for the local RAG project."""

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

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def main() -> None:
    st.set_page_config(page_title="Local RAG", layout="wide")
    _init_state()

    base_settings = load_settings()
    
    # Проверка атрибутов для диагностики
    has_use_reranker = hasattr(base_settings, 'use_reranker')
    has_reranker_model = hasattr(base_settings, 'reranker_model')
    has_rerank_candidates = hasattr(base_settings, 'rerank_candidates')
    
    if not (has_use_reranker and has_reranker_model and has_rerank_candidates):
        import sys
        print(f"WARNING: Settings объект имеет неполные атрибуты!", file=sys.stderr)
        print(f"  - use_reranker: {has_use_reranker}", file=sys.stderr)
        print(f"  - reranker_model: {has_reranker_model}", file=sys.stderr)
        print(f"  - rerank_candidates: {has_rerank_candidates}", file=sys.stderr)

    st.title("Local RAG")
    st.caption("Загрузка документов, FAISS-индексация, reranking и чат с OpenRouter LLM.")

    with st.sidebar:
        st.header("Настройки")
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
        use_reranker = getattr(base_settings, 'use_reranker', True)
        st.caption(f"Reranker: {'on' if use_reranker else 'off'}")
        if use_reranker:
            reranker_model = getattr(base_settings, 'reranker_model', 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
            st.caption(reranker_model)

        if st.button("Применить настройки", use_container_width=True):
            st.session_state.chat_session = None
            st.session_state.chat_session_signature = None
            st.success("Настройки применены")

        if st.button("Очистить чат", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_context = []
            st.rerun()

        if st.button("Сбросить состояние UI", use_container_width=True):
            _reset_ui_state()
            st.rerun()

    settings = _settings_from_state(base_settings)
    raw_data_dir = settings.raw_data_dir
    index_dir = settings.vector_store_dir

    _render_status(settings, raw_data_dir, index_dir)
    _render_upload_and_index(settings, raw_data_dir, index_dir)
    _render_chat(settings, index_dir)


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


def _settings_from_state(settings: Settings) -> Settings:
    return replace(
        settings,
        top_k=int(st.session_state.top_k),
        max_context_chars=int(st.session_state.max_context_chars),
        temperature=float(st.session_state.temperature),
    )


def _render_status(settings: Settings, raw_data_dir: Path, index_dir: Path) -> None:
    documents = _list_documents(raw_data_dir)
    index_ready = _index_exists(index_dir)
    st.session_state.index_ready = index_ready

    st.subheader("Статус")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Индекс", "загружен" if index_ready else "не создан")
    col2.metric("Документы", str(len(documents)))
    col3.metric("LLM model", settings.openrouter_model)
    col4.metric("Reranker", "on" if getattr(settings, 'use_reranker', True) else "off")

    if not settings.openrouter_api_key:
        st.warning("OPENROUTER_API_KEY не найден. Добавьте ключ в .env перед запуском чата.")


def _render_upload_and_index(settings: Settings, raw_data_dir: Path, index_dir: Path) -> None:
    st.subheader("Документы")
    if st.session_state.notice:
        st.success(st.session_state.notice)
        st.session_state.notice = None

    uploaded_files = st.file_uploader(
        "Загрузите PDF или DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        _save_uploaded_files(uploaded_files, raw_data_dir)

    if st.session_state.uploaded_files:
        st.caption("Файлы в этой сессии: " + ", ".join(st.session_state.uploaded_files))

    _render_document_list(raw_data_dir, index_dir)

    clear_index = st.checkbox("Также удалить FAISS-индекс", value=True)
    if st.button("Очистить все загруженные документы"):
        try:
            removed_documents = _clear_uploaded_documents(raw_data_dir)
            removed_index_files = _clear_index_files(index_dir) if clear_index else 0
            _reset_index_dependent_state()
            st.session_state.uploaded_files = []
            st.session_state.index_ready = _index_exists(index_dir)
            st.session_state.notice = (
                f"Удалено документов: {removed_documents}. "
                f"Файлов индекса: {removed_index_files}."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Ошибка очистки: {exc}")

    if st.button("Проиндексировать документы", type="primary"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(stage: str, current: int, total: int) -> None:
                if total > 0:
                    progress = current / total
                    progress_bar.progress(progress, text=f"{stage}: {current}/{total}")
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
            st.success(f"Индекс обновлен. Chunks: {count}")
        except Exception as exc:
            st.session_state.index_ready = False
            st.error(f"Ошибка индексации: {exc}")


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


def _render_document_list(raw_data_dir: Path, index_dir: Path) -> None:
    documents = _list_documents(raw_data_dir)
    if not documents:
        st.caption("Загруженных документов пока нет.")
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
                st.session_state.notice = (
                    f"Документ удален: {relative_path}. "
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


def _render_chat(settings: Settings, index_dir: Path) -> None:
    st.subheader("Чат")
    if not _index_exists(index_dir):
        st.info("Сначала загрузите документы и создайте индекс.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Вопрос", placeholder="О чем этот документ?")
        submitted = st.form_submit_button("Отправить", type="primary")

    if not submitted:
        _render_context(st.session_state.last_context)
        return
    if not question or not question.strip():
        st.warning("Введите вопрос перед отправкой.")
        _render_context(st.session_state.last_context)
        return

    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        session = _get_chat_session(settings, index_dir)
        with st.spinner("Ищу контекст, ранжирую источники и запрашиваю LLM..."):
            response = session.stream(question, top_k=int(st.session_state.top_k))
    except MissingAPIKeyError as exc:
        _show_llm_error("Ошибка конфигурации", exc)
        return
    except LLMTimeoutError as exc:
        _show_llm_error("OpenRouter timeout", exc)
        return
    except InvalidModelError as exc:
        _show_llm_error("Модель недоступна", exc)
        return
    except LLMClientError as exc:
        _show_llm_error("Ошибка LLM", exc)
        return
    except FileNotFoundError:
        st.error("Индекс не найден. Сначала выполните индексацию.")
        return
    except Exception as exc:
        st.error(f"Ошибка обработки вопроса: {exc}")
        return

    st.session_state.last_context = response.results

    with st.chat_message("assistant"):
        answer_box = st.empty()
        answer_box.markdown("печатает...")
        answer = ""
        try:
            for token in response.chunks:
                answer += token
                answer_box.markdown(answer + "▌")
            answer_box.markdown(answer)
        except MissingAPIKeyError as exc:
            _show_llm_error("Ошибка конфигурации", exc)
            return
        except LLMTimeoutError as exc:
            _show_llm_error("OpenRouter timeout", exc)
            return
        except InvalidModelError as exc:
            _show_llm_error("Модель недоступна", exc)
            return
        except LLMClientError as exc:
            _show_llm_error("Ошибка LLM", exc)
            return

    st.session_state.messages.append({"role": "assistant", "content": answer})
    _render_context(response.results)


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


def _show_llm_error(title: str, exc: Exception) -> None:
    st.error(f"{title}: {exc}")
    st.session_state.messages.append({"role": "assistant", "content": f"{title}: {exc}"})


def _render_context(results) -> None:
    st.subheader("Источники")
    if not results:
        st.caption("Источники не найдены.")
        return

    for index, citation in enumerate(build_citations(results), start=1):
        page = citation.page or "n/a"
        with st.container(border=True):
            st.markdown(f"**[{citation.label}] {citation.source}**")
            col_page, col_chunk, col_score = st.columns(3)
            col_page.caption(f"Page: {page}")
            col_chunk.caption(f"Chunk: {citation.chunk_index}")
            col_score.caption(f"Relevance: {citation.score:.4f}")
            with st.expander("Показать фрагмент", expanded=index == 1):
                st.write(citation.text)


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


def _reset_index_dependent_state() -> None:
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
