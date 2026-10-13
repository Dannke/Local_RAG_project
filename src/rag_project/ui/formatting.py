"""Pure display helpers for the Streamlit UI (no Streamlit imports)."""

from __future__ import annotations

import math
from pathlib import Path

_FILE_ICONS = {
    ".pdf": "📄",
    ".docx": "📝",
    ".doc": "📝",
    ".txt": "📋",
    ".md": "📖",
}

_FILE_TYPES = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".doc": "DOC",
    ".txt": "Text",
    ".md": "Markdown",
}

# Relevance thresholds and their styling (display only — does not affect retrieval).
_RELEVANCE_TIERS = (
    (0.80, "Excellent Match", "#66bb6a"),  # green
    (0.60, "Good Match", "#9ccc65"),  # light green
    (0.40, "Moderate Match", "#ffa726"),  # amber
    (0.00, "Weak Match", "#ef5350"),  # red
)


def file_icon(file_extension: str) -> str:
    return _FILE_ICONS.get(file_extension.lower(), "📎")


def file_type(file_extension: str) -> str:
    return _FILE_TYPES.get(file_extension.lower(), "File")


def human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def relevance_for(score: object) -> tuple[int, str, str] | None:
    """Map a raw score to a (percent, label, color) triple for display only.

    Returns None when the score is not a usable number (None, NaN, non-numeric),
    so the relevance section can be hidden entirely. The backend score itself
    is never modified — only displayed and clamped to a 0–100% bar.
    """
    try:
        numeric = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    percent = max(0, min(int(round(numeric * 100)), 100))
    for threshold, label, color in _RELEVANCE_TIERS:
        if numeric >= threshold:
            return percent, label, color
    _, label, color = _RELEVANCE_TIERS[-1]
    return percent, label, color


def citation_filename(source: str) -> str:
    """Return a readable filename from a citation source path/id."""
    name = Path(source).name if source else source
    return name or "Unknown source"


def citation_preview(text: str, limit: int = 260) -> str:
    """Short preview of the chunk text, whitespace collapsed for compactness."""
    if not text:
        return ""
    snippet = " ".join(text.split())
    if len(snippet) <= limit:
        return snippet
    return snippet[:limit].rstrip() + "…"
