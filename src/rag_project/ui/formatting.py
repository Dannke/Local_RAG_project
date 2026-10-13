"""Pure display helpers for the Streamlit UI (no Streamlit imports)."""

from __future__ import annotations

import math
from pathlib import Path

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


def file_type(file_extension: str) -> str:
    return _FILE_TYPES.get(file_extension.lower(), "File")


def human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def relevance_for(
    score: object,
    score_min: float | None = None,
    score_max: float | None = None,
) -> tuple[int, str, str] | None:
    """Map a raw score to a (percent, label, color) triple for display only.

    The cross-encoder returns raw logits (roughly -10…10), well ranked but badly
    calibrated, so scores are normalized relative to the other sources of the
    same answer (best = 100%, worst = 0%) when ``score_min < score_max``, and
    squashed with a sigmoid otherwise.

    Returns None when the score is not a usable number (None, NaN, non-numeric),
    so the relevance section can be hidden entirely. The backend score itself
    is never modified.
    """
    try:
        numeric = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    if score_min is not None and score_max is not None and score_max > score_min:
        probability = (numeric - score_min) / (score_max - score_min)
    else:
        probability = 1.0 / (1.0 + math.exp(-numeric))
    percent = max(0, min(int(round(probability * 100)), 100))
    for threshold, label, color in _RELEVANCE_TIERS:
        if probability >= threshold:
            return percent, label, color
    _, label, color = _RELEVANCE_TIERS[-1]
    return percent, label, color


# Display-only names for the tiers above: (CSS modifier, Russian label).
_RELEVANCE_DISPLAY = {
    "Excellent Match": ("excellent", "высокая"),
    "Good Match": ("good", "хорошая"),
    "Moderate Match": ("moderate", "средняя"),
    "Weak Match": ("weak", "слабая"),
}


def relevance_tier(label: str) -> tuple[str, str]:
    """(CSS modifier, Russian label) for a tier label returned by ``relevance_for``."""
    return _RELEVANCE_DISPLAY.get(label, _RELEVANCE_DISPLAY["Weak Match"])


def plural_ru(count: int, one: str, few: str, many: str) -> str:
    """``count`` with the Russian noun form: 1 чат, 2 чата, 5 чатов, 21 чат."""
    tail = abs(count) % 100
    if 11 <= tail <= 14:
        form = many
    elif tail % 10 == 1:
        form = one
    elif 2 <= tail % 10 <= 4:
        form = few
    else:
        form = many
    return f"{count} {form}"


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
