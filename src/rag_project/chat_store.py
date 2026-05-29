"""File-backed chat sessions for the Streamlit RAG UI."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChatMeta:
    id: str
    title: str
    created_at: str
    updated_at: str
    path: Path


def chats_root(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "chats"


def initialize_chats(root: str | Path) -> list[ChatMeta]:
    chat_root = Path(root)
    chat_root.mkdir(parents=True, exist_ok=True)
    chats = list_chats(chat_root)
    if chats:
        return chats
    return [create_chat(chat_root)]


def list_chats(root: str | Path) -> list[ChatMeta]:
    chat_root = Path(root)
    if not chat_root.exists():
        return []

    chats: list[ChatMeta] = []
    for chat_dir in sorted(chat_root.iterdir()):
        if not chat_dir.is_dir():
            continue
        meta = _read_meta(chat_dir)
        if meta is not None:
            chats.append(meta)
    return sorted(chats, key=lambda chat: chat.created_at)


def create_chat(
    root: str | Path,
    title: str | None = None,
    now: datetime | None = None,
) -> ChatMeta:
    chat_root = Path(root)
    chat_root.mkdir(parents=True, exist_ok=True)
    current_time = now or datetime.now().astimezone()
    chat_id = _next_chat_id(chat_root)
    chat_dir = chat_root / chat_id
    get_chat_documents_dir(chat_dir).mkdir(parents=True, exist_ok=True)
    get_chat_index_dir(chat_dir).mkdir(parents=True, exist_ok=True)

    timestamp = current_time.isoformat(timespec="seconds")
    meta = ChatMeta(
        id=chat_id,
        title=title or current_time.strftime("Chat %d.%m %H:%M"),
        created_at=timestamp,
        updated_at=timestamp,
        path=chat_dir,
    )
    _write_meta(meta)
    save_messages(chat_dir, [])
    return meta


def delete_chat(root: str | Path, chat_id: str) -> None:
    chat_root = Path(root)
    chat_dir = _safe_chat_dir(chat_root, chat_id)
    if chat_dir.exists():
        shutil.rmtree(chat_dir)


def rename_chat(root: str | Path, chat_id: str, title: str) -> ChatMeta:
    chat_root = Path(root)
    chat_dir = _safe_chat_dir(chat_root, chat_id)
    meta = _require_meta(chat_dir)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    renamed = ChatMeta(
        id=meta.id,
        title=title.strip() or meta.title,
        created_at=meta.created_at,
        updated_at=now,
        path=meta.path,
    )
    _write_meta(renamed)
    return renamed


def load_messages(chat_dir: str | Path) -> list[dict[str, str]]:
    path = get_chat_messages_path(chat_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []

    messages: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


def save_messages(chat_dir: str | Path, messages: list[dict[str, str]]) -> None:
    path = get_chat_messages_path(chat_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_chat_documents_dir(chat_dir: str | Path) -> Path:
    return Path(chat_dir) / "documents"


def get_chat_index_dir(chat_dir: str | Path) -> Path:
    return Path(chat_dir) / "index"


def get_chat_messages_path(chat_dir: str | Path) -> Path:
    return Path(chat_dir) / "messages.json"


def get_chat_meta_path(chat_dir: str | Path) -> Path:
    return Path(chat_dir) / "meta.json"


def ensure_chat_layout(chat_dir: str | Path) -> None:
    get_chat_documents_dir(chat_dir).mkdir(parents=True, exist_ok=True)
    get_chat_index_dir(chat_dir).mkdir(parents=True, exist_ok=True)
    messages_path = get_chat_messages_path(chat_dir)
    if not messages_path.exists():
        save_messages(chat_dir, [])


def _next_chat_id(root: Path) -> str:
    highest = 0
    for path in root.glob("chat_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("chat_")
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"chat_{highest + 1:03d}"


def _read_meta(chat_dir: Path) -> ChatMeta | None:
    meta_path = get_chat_meta_path(chat_dir)
    if not meta_path.exists():
        return None
    try:
        payload: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    chat_id = payload.get("id")
    title = payload.get("title")
    created_at = payload.get("created_at")
    updated_at = payload.get("updated_at", created_at)
    if not all(isinstance(value, str) for value in (chat_id, title, created_at, updated_at)):
        return None
    return ChatMeta(
        id=str(chat_id),
        title=str(title),
        created_at=str(created_at),
        updated_at=str(updated_at),
        path=chat_dir,
    )


def _require_meta(chat_dir: Path) -> ChatMeta:
    meta = _read_meta(chat_dir)
    if meta is None:
        raise FileNotFoundError(f"Chat metadata not found: {chat_dir}")
    return meta


def _write_meta(meta: ChatMeta) -> None:
    meta.path.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": meta.id,
        "title": meta.title,
        "created_at": meta.created_at,
        "updated_at": meta.updated_at,
    }
    get_chat_meta_path(meta.path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _safe_chat_dir(root: Path, chat_id: str) -> Path:
    if "/" in chat_id or "\\" in chat_id or chat_id in {"", ".", ".."}:
        raise RuntimeError(f"Invalid chat id: {chat_id}")
    resolved_root = root.resolve()
    chat_dir = (root / chat_id).resolve()
    if not chat_dir.is_relative_to(resolved_root):
        raise RuntimeError(f"Refusing to access chat outside {resolved_root}: {chat_dir}")
    return chat_dir
