"""Documents page: knowledge-base dashboard, upload, document cards, index management."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from rag_project.application.indexing import remove_document
from rag_project.bootstrap import ingest_to_faiss
from rag_project.config import Settings
from rag_project.infrastructure.document_directory import UnsupportedDocumentError
from rag_project.infrastructure.loaders import SUPPORTED_EXTENSIONS
from rag_project.ui.formatting import file_type, human_size, plural_ru
from rag_project.ui.state import (
    bump_uploader_version,
    invalidate_chat_session,
    reset_index_dependent_state,
    uploader_key,
)
from rag_project.ui.theme import (
    empty_state_html,
    kpi_card_html,
    render_html,
    render_page_header,
    render_section_title,
)
from rag_project.ui.workspace import ChatWorkspace

_BYTES_PER_MB = 1024 * 1024


def render_documents_page(settings: Settings, workspace: ChatWorkspace) -> None:
    render_page_header("Документы", f"База знаний чата «{html.escape(workspace.chat.title)}»")
    if st.session_state.notice:
        st.success(st.session_state.notice, icon=":material/check_circle:")
        st.session_state.notice = None
    documents = sorted(workspace.documents.list_files())
    _render_dashboard(workspace, documents)
    _render_upload_section(workspace)
    _render_document_cards(workspace, documents)
    _render_index_management(settings, workspace, len(documents))


# ─────────────────────────── Knowledge Base Dashboard ───────────────


def _render_dashboard(workspace: ChatWorkspace, documents: list[Path]) -> None:
    index_ready = workspace.index.exists()
    stats = workspace.index.stats()
    index_size_mb = stats.size_bytes / _BYTES_PER_MB
    chunks = stats.chunk_count if stats.chunk_count is not None else "—"

    render_html(
        '<div class="kpi-grid">'
        + kpi_card_html("Документы", len(documents))
        + kpi_card_html("Фрагменты", chunks)
        + kpi_card_html(
            "Индекс",
            "Готов" if index_ready else "Не создан",
            tone="ok" if index_ready else "warn",
        )
        + kpi_card_html("Размер индекса", f"{index_size_mb:.1f} МБ" if index_size_mb > 0 else "—")
        + "</div>"
    )


# ─────────────────────────── Upload Section ───────────────────────────


def _render_upload_section(workspace: ChatWorkspace) -> None:
    render_section_title("Загрузка", "PDF · DOCX · TXT · MD")

    with st.container(key="upload-zone"):
        uploaded_files = st.file_uploader(
            "Загрузите PDF, DOCX, TXT или MD",
            type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=uploader_key(workspace.chat.id),
        )
    if uploaded_files:
        _save_uploaded_files(uploaded_files, workspace)

    if st.session_state.uploaded_files:
        files = "".join(
            f'<li class="chip">{html.escape(name)}</li>' for name in st.session_state.uploaded_files
        )
        render_html(
            '<div class="session-files"><span>Загружено в этой сессии:</span>'
            f'<ul class="chip-list">{files}</ul></div>'
        )


def _save_uploaded_files(uploaded_files: list, workspace: ChatWorkspace) -> None:
    saved: list[str] = []
    errors: list[str] = []
    for uploaded_file in uploaded_files:
        try:
            saved.append(workspace.documents.save(uploaded_file.name, uploaded_file.getbuffer()))
        except UnsupportedDocumentError as exc:
            errors.append(str(exc))

    if saved:
        st.session_state.uploaded_files = sorted(set(st.session_state.uploaded_files + saved))
        st.success("Загружено: " + ", ".join(saved), icon=":material/upload_file:")
    for error in errors:
        st.error(error)


# ─────────────────────────── Document Cards ───────────────────────


def _render_document_cards(workspace: ChatWorkspace, documents: list[Path]) -> None:
    render_section_title("Файлы", plural_ru(len(documents), "документ", "документа", "документов"))

    if not documents:
        render_html(
            empty_state_html(
                "Документов пока нет",
                "Перетащите файлы в зону загрузки выше, затем постройте индекс.",
            )
        )
        return

    for position, document_path in enumerate(documents):
        _render_document_card(position, document_path, workspace)


def _render_document_card(position: int, document_path: Path, workspace: ChatWorkspace) -> None:
    relative_path = workspace.documents.relative_path(document_path)
    name = Path(relative_path).name
    extension = document_path.suffix.lower()

    with st.container(
        key=f"doc-card-{position}", horizontal=True, vertical_alignment="center", gap="small"
    ):
        render_html(
            '<div class="doc-card">'
            f'<span class="doc-type" aria-hidden="true">'
            f"{html.escape(extension.lstrip('.')[:4]) or 'file'}</span>"
            '<div class="doc-body">'
            f'<div class="doc-name">{html.escape(name)}</div>'
            f'<div class="doc-meta">{file_type(extension)} · '
            f"{human_size(document_path.stat().st_size)}</div>"
            "</div></div>"
        )
        if st.button(
            ":material/refresh:",
            key=f"reindex_{relative_path}",
            type="tertiary",
            help="Переиндексировать: индекс обновляется для всех документов сразу",
        ):
            st.info("Нажмите «Индексировать», чтобы обновить индекс.")
        if st.button(
            ":material/delete:",
            key=f"delete_{relative_path}",
            type="tertiary",
            help=f"Удалить {name}",
        ):
            _confirm_delete_document(document_path, workspace)


@st.dialog("Удалить документ?")
def _confirm_delete_document(document_path: Path, workspace: ChatWorkspace) -> None:
    relative_path = workspace.documents.relative_path(document_path)
    render_html(
        f"<p>Файл <b>{html.escape(Path(relative_path).name)}</b> будет удалён, "
        "а его фрагменты — убраны из индекса этого чата.</p>"
    )
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Отмена", key="cancel_delete_document"):
            st.rerun()
        if st.button("Удалить документ", key="confirm_delete_document", type="primary"):
            removed_chunks = _delete_document(document_path, workspace)
            if removed_chunks is not None:
                bump_uploader_version(workspace.chat.id)
                index_note = (
                    "Его фрагменты удалены из индекса."
                    if removed_chunks
                    else "Фрагментов в индексе не найдено; "
                    "если он был проиндексирован раньше, нажмите «Пересобрать»."
                )
                st.session_state.notice = f"Документ удалён: {relative_path}. {index_note}"
                st.rerun()


def _delete_document(path: Path, workspace: ChatWorkspace) -> int | None:
    """Delete the file, then remove only its vectors from the index.

    Returns the number of chunks removed, or None (after showing the error)
    if either step failed.
    """
    try:
        workspace.documents.delete(path)
    except Exception as exc:
        st.error(f"Ошибка удаления файла: {exc}")
        return None
    try:
        removed = remove_document(workspace.index, workspace.documents.relative_path(path))
    except Exception as exc:
        st.error(f"Ошибка удаления из индекса: {exc}")
        return None
    st.session_state.uploaded_files = [
        name for name in st.session_state.uploaded_files if name != path.name
    ]
    reset_index_dependent_state()
    return removed


# ─────────────────────────── Index Management ──────────────────────


def _render_index_management(
    settings: Settings, workspace: ChatWorkspace, document_count: int
) -> None:
    render_section_title("Индекс", "FAISS · обновляется по кнопке")

    with st.container(horizontal=True, gap="small", key="index-actions"):
        index_clicked = st.button(
            "Индексировать",
            type="primary",
            icon=":material/build:",
            help="Добавить в индекс новые и изменённые документы",
        )
        rebuild_clicked = st.button(
            "Пересобрать",
            icon=":material/autorenew:",
            help="Построить индекс заново из всех документов",
        )
        if st.button("Очистить индекс", icon=":material/delete_sweep:"):
            _confirm_clear_index(workspace)
        if st.button("Удалить все документы", icon=":material/folder_delete:"):
            _confirm_clear_documents(workspace, document_count)

    if index_clicked:
        _run_indexing(
            settings,
            workspace,
            incremental=True,
            status="Индексация…",
            success="Индекс обновлён",
            failure="Ошибка индексации",
        )
    if rebuild_clicked:
        _run_indexing(
            settings,
            workspace,
            incremental=False,
            status="Пересборка индекса…",
            success="Индекс пересобран",
            failure="Ошибка пересборки",
        )


@st.dialog("Очистить индекс?")
def _confirm_clear_index(workspace: ChatWorkspace) -> None:
    render_html(
        "<p>Файлы индекса будут удалены, документы останутся. "
        "Чтобы снова задавать вопросы, индекс нужно будет построить заново.</p>"
    )
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Отмена", key="cancel_clear_index"):
            st.rerun()
        if st.button("Очистить индекс", key="confirm_clear_index", type="primary"):
            try:
                removed = workspace.index.clear()
                reset_index_dependent_state()
                st.session_state.notice = f"Индекс очищен. Удалено файлов: {removed}"
                st.rerun()
            except Exception as exc:
                st.error(f"Ошибка очистки индекса: {exc}")


@st.dialog("Удалить все документы?")
def _confirm_clear_documents(workspace: ChatWorkspace, count: int) -> None:
    render_html(
        f"<p>Из чата будет удалено: {plural_ru(count, 'документ', 'документа', 'документов')}. "
        "Это действие нельзя отменить.</p>"
    )
    clear_index = st.checkbox("Также удалить FAISS-индекс", value=True)
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Отмена", key="cancel_clear_documents"):
            st.rerun()
        if st.button("Удалить всё", key="confirm_clear_documents", type="primary"):
            try:
                removed_documents = workspace.documents.clear()
                removed_index_files = workspace.index.clear() if clear_index else 0
                reset_index_dependent_state()
                st.session_state.uploaded_files = []
                bump_uploader_version(workspace.chat.id)
                st.session_state.notice = (
                    f"Удалено документов: {removed_documents}. "
                    f"Файлов индекса: {removed_index_files}."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Ошибка удаления: {exc}")


def _run_indexing(
    settings: Settings,
    workspace: ChatWorkspace,
    *,
    incremental: bool,
    status: str,
    success: str,
    failure: str,
) -> None:
    """Run ingestion with a live progress bar inside a status panel.

    A rebuild builds the new index in memory and only replaces the saved one on
    success, so a failed rebuild keeps the previous index usable.
    """
    with st.status(status, expanded=True) as status_panel:
        try:
            progress_bar = st.progress(0, text=status)

            def update_progress(stage: str, current: int, total: int) -> None:
                if total > 0:
                    progress_bar.progress(current / total, text=f"{stage}: {current}/{total}")
                else:
                    status_panel.update(label=stage)

            count = ingest_to_faiss(
                input_dir=workspace.documents_dir,
                index_dir=workspace.index_dir,
                settings=settings,
                incremental=incremental,
                progress_callback=update_progress,
            )
            progress_bar.progress(1.0, text="Готово")
            invalidate_chat_session()
            status_panel.update(label=success, state="complete")
        except Exception as exc:
            status_panel.update(label=failure, state="error")
            st.error(f"{failure}: {exc}")
            return
    # Rerun so the dashboard above shows the new index; the notice carries the result.
    st.session_state.notice = f"{success}. Фрагментов: {count}"
    st.rerun()
