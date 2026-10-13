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
from rag_project.ui.formatting import file_icon, file_type, human_size
from rag_project.ui.state import (
    bump_uploader_version,
    invalidate_chat_session,
    reset_index_dependent_state,
    uploader_key,
)
from rag_project.ui.theme import (
    ACCENT_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
    metric_card_html,
    render_html,
    render_section_divider,
)
from rag_project.ui.workspace import ChatWorkspace

_BYTES_PER_MB = 1024 * 1024


def render_documents_page(settings: Settings, workspace: ChatWorkspace) -> None:
    st.markdown("### 📚 Документы")
    _render_dashboard(workspace)
    render_section_divider()
    _render_upload_section(workspace)
    render_section_divider()
    _render_document_cards(workspace)
    render_section_divider()
    _render_index_management(settings, workspace)


# ─────────────────────────── Knowledge Base Dashboard ───────────────


def _render_dashboard(workspace: ChatWorkspace) -> None:
    st.markdown("#### Knowledge Base")

    documents = workspace.documents.list_files()
    index_ready = workspace.index.exists()
    stats = workspace.index.stats()
    index_status = "Ready ✓" if index_ready else "Not created"
    index_size_mb = stats.size_bytes / _BYTES_PER_MB
    chunks = stats.chunk_count if stats.chunk_count is not None else "N/A"

    cards = (
        metric_card_html("📄 Documents", len(documents)),
        metric_card_html("🔗 Chunks", chunks),
        metric_card_html(
            "⚙️ Index",
            index_status,
            color=SUCCESS_COLOR if index_ready else WARNING_COLOR,
            font_size="0.95rem",
        ),
        metric_card_html(
            "💾 Index Size",
            f"{index_size_mb:.1f} MB" if index_size_mb > 0 else "N/A",
            color=ACCENT_COLOR,
            font_size="1.2rem",
        ),
    )
    for column, card in zip(st.columns(4, gap="small"), cards, strict=True):
        with column:
            render_html(card)


# ─────────────────────────── Upload Section ───────────────────────────


def _render_upload_section(workspace: ChatWorkspace) -> None:
    st.markdown("#### Upload Documents")

    col_uploader, col_badge = st.columns([4, 1])

    with col_uploader:
        uploaded_files = st.file_uploader(
            "Загрузите PDF, DOCX, TXT или MD",
            type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=uploader_key(workspace.chat.id),
        )
        if uploaded_files:
            _save_uploaded_files(uploaded_files, workspace)

    with col_badge:
        render_html(
            '<div style="background:#7c4dff;color:white;padding:8px 12px;border-radius:8px;'
            'text-align:center;font-size:0.75rem;font-weight:700;">'
            "PDF · DOCX · TXT · MD"
            "</div>"
        )

    if st.session_state.notice:
        st.success(st.session_state.notice)
        st.session_state.notice = None

    if st.session_state.uploaded_files:
        files_list = ", ".join(html.escape(name) for name in st.session_state.uploaded_files)
        render_html(
            '<div style="font-size:0.82rem;color:#8b92a8;margin:0.8rem 0;'
            'background:#1a1f2e;padding:8px 12px;border-radius:6px;">'
            f"📋 Session files: {files_list}</div>"
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
        st.success("Loaded: " + ", ".join(saved))
    for error in errors:
        st.error(error)


# ─────────────────────────── Document Cards ───────────────────────


def _render_document_cards(workspace: ChatWorkspace) -> None:
    st.markdown("#### Documents")

    documents = workspace.documents.list_files()
    if not documents:
        _render_empty_state()
        return

    for document_path in sorted(documents):
        _render_document_card(document_path, workspace)


def _render_empty_state() -> None:
    render_html(
        '<div style="text-align:center;padding:3rem 2rem;color:#5a6078;">'
        '<div style="font-size:3rem;margin-bottom:1rem;">📂</div>'
        '<div style="font-size:1.1rem;font-weight:500;margin-bottom:0.5rem;">'
        "No documents uploaded</div>"
        '<div style="font-size:0.9rem;color:#8b92a8;">'
        "Upload PDF, DOCX, TXT or MD files to start building your knowledge base."
        "</div>"
        "</div>"
    )


def _render_document_card(document_path: Path, workspace: ChatWorkspace) -> None:
    relative_path = workspace.documents.relative_path(document_path)
    extension = document_path.suffix.lower()

    with st.columns(1)[0]:
        render_html(
            '<div style="background:#1a1f2e;border:1px solid #2a3040;border-radius:10px;'
            'padding:16px;margin-bottom:10px;transition:all 0.2s;">'
            '<div style="display:flex;align-items:flex-start;justify-content:space-between;'
            'gap:12px;">'
            '<div style="flex:1;">'
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
            f'<span style="font-size:1.5rem;">{file_icon(extension)}</span>'
            "<div>"
            '<div style="font-weight:600;font-size:0.95rem;color:#e0e0e0;">'
            f"{html.escape(Path(relative_path).name)}</div>"
            f'<div style="font-size:0.8rem;color:#8b92a8;">{file_type(extension)}</div>'
            "</div>"
            "</div>"
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:0.8rem;">'
            '<div style="color:#8b92a8;"><b>Size:</b> '
            f"{human_size(document_path.stat().st_size)}</div>"
            f'<div style="color:#8b92a8;"><b>Chat:</b> {html.escape(workspace.chat.title)}</div>'
            "</div>"
            "</div>"
            "</div>"
        )

        col_delete, col_reindex = st.columns(2)

        with col_delete:
            if st.button(
                "🗑 Delete",
                key=f"delete_{relative_path}",
                use_container_width=True,
                type="secondary",
            ):
                removed_chunks = _delete_document(document_path, workspace)
                if removed_chunks is not None:
                    bump_uploader_version(workspace.chat.id)
                    index_note = (
                        "Its vectors were removed from the index."
                        if removed_chunks
                        else "No indexed chunks were found for it; "
                        "run 'Rebuild Index' if it was indexed before."
                    )
                    st.session_state.notice = f"Document deleted: {relative_path}. {index_note}"
                    st.rerun()

        with col_reindex:
            if st.button(
                "🔄 Reindex",
                key=f"reindex_{relative_path}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.notice = (
                    "Please use 'Index Documents' button to reindex all documents."
                )
                st.info("Run 'Index Documents' to rebuild the index.")


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


def _render_index_management(settings: Settings, workspace: ChatWorkspace) -> None:
    st.markdown("#### ⚙️ Index Management")

    col_status, col_docs = st.columns(2)
    col_status.metric("Status", "Ready ✓" if workspace.index.exists() else "Not created")
    col_docs.metric("Documents loaded", str(len(workspace.documents.list_files())))

    col_index, col_clear, col_rebuild = st.columns(3)

    with col_index:
        if st.button("🔨 Index Documents", type="primary", use_container_width=True):
            _run_indexing(
                settings,
                workspace,
                incremental=True,
                status="Initializing indexing...",
                success="Index updated",
                failure="Indexing error",
            )

    with col_clear:
        if st.button("🗑 Clear Index", use_container_width=True):
            try:
                removed = workspace.index.clear()
                reset_index_dependent_state()
                st.success(f"Index cleared. Files removed: {removed}")
                st.rerun()
            except Exception as exc:
                st.error(f"Error clearing index: {exc}")

    with col_rebuild:
        if st.button("🔄 Rebuild Index", use_container_width=True):
            _run_indexing(
                settings,
                workspace,
                incremental=False,
                status="Rebuilding index...",
                success="Index rebuilt",
                failure="Rebuild error",
            )

    st.markdown("**Clear all documents**")
    clear_index = st.checkbox("Also delete FAISS index", value=True)
    if st.button("🗑 Clear all uploaded documents", use_container_width=True):
        try:
            removed_documents = workspace.documents.clear()
            removed_index_files = workspace.index.clear() if clear_index else 0
            reset_index_dependent_state()
            st.session_state.uploaded_files = []
            bump_uploader_version(workspace.chat.id)
            st.session_state.notice = (
                f"Documents removed: {removed_documents}. Index files: {removed_index_files}."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Clear error: {exc}")


def _run_indexing(
    settings: Settings,
    workspace: ChatWorkspace,
    *,
    incremental: bool,
    status: str,
    success: str,
    failure: str,
) -> None:
    """Run ingestion with a live progress bar.

    A rebuild builds the new index in memory and only replaces the saved one on
    success, so a failed rebuild keeps the previous index usable.
    """
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(stage: str, current: int, total: int) -> None:
            if total > 0:
                progress_bar.progress(current / total, text=f"{stage}: {current}/{total}")
            else:
                status_text.text(stage)

        status_text.text(status)
        count = ingest_to_faiss(
            input_dir=workspace.documents_dir,
            index_dir=workspace.index_dir,
            settings=settings,
            incremental=incremental,
            progress_callback=update_progress,
        )
        progress_bar.progress(1.0, text="Complete!")
        invalidate_chat_session()
        st.success(f"{success}. Chunks: {count}")
    except Exception as exc:
        st.error(f"{failure}: {exc}")
