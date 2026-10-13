"""Per-chat workspace: the chat, its source documents and its vector index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_project.infrastructure.chat_store import (
    ChatMeta,
    get_chat_documents_dir,
    get_chat_index_dir,
)
from rag_project.infrastructure.document_directory import DocumentDirectory
from rag_project.infrastructure.index_storage import FaissIndexRepository


@dataclass(frozen=True)
class ChatWorkspace:
    chat: ChatMeta
    documents: DocumentDirectory
    index: FaissIndexRepository

    @classmethod
    def for_chat(cls, chat: ChatMeta) -> ChatWorkspace:
        return cls(
            chat=chat,
            documents=DocumentDirectory(get_chat_documents_dir(chat.path)),
            index=FaissIndexRepository(get_chat_index_dir(chat.path)),
        )

    @property
    def documents_dir(self) -> Path:
        return self.documents.root

    @property
    def index_dir(self) -> Path:
        return self.index.index_dir
