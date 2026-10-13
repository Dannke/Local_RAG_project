"""File-system helpers guarding against operations outside a managed root."""

from __future__ import annotations

from pathlib import Path


def safe_unlink(path: Path, root: Path) -> None:
    """Delete ``path`` if it is a file inside ``root``; refuse anything outside."""
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise RuntimeError(f"Refusing to delete outside {resolved_root}: {resolved_path}")
    if resolved_path.is_file():
        resolved_path.unlink()
