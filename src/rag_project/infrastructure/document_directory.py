"""A directory of source documents: scanning, loading and file management."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag_project.domain.manifest import SourceFileRecord
from rag_project.domain.models import Document
from rag_project.infrastructure.filesystem import safe_unlink
from rag_project.infrastructure.loaders import (
    SUPPORTED_EXTENSIONS,
    load_document_file,
    load_documents,
)


class UnsupportedDocumentError(ValueError):
    """Raised when a file type has no loader."""


class DocumentDirectory:
    """Source documents stored under one root directory.

    Implements :class:`~rag_project.application.ports.DocumentSource` for
    indexing and offers the file management used by the UI.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- DocumentSource ------------------------------------------------------

    def scan(self) -> dict[str, SourceFileRecord]:
        return scan_source_files(self.root)

    def load_all(self) -> list[Document]:
        return load_documents(self.root)

    def load(self, relative_path: str) -> list[Document]:
        return load_document_file(self.root / relative_path, root=self.root)

    # -- File management ------------------------------------------------------

    def list_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return [path for path in self.root.rglob("*") if is_supported_file(path)]

    def relative_path(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def save(self, filename: str, data: bytes | bytearray | memoryview) -> str:
        """Store an uploaded file under its base name; return that name."""
        name = Path(filename).name
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedDocumentError(f"{name}: unsupported format")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / name).write_bytes(data)
        return name

    def delete(self, path: Path) -> None:
        safe_unlink(path, self.root)

    def clear(self) -> int:
        removed = 0
        for path in self.list_files():
            self.delete(path)
            removed += 1
        return removed


def is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def scan_source_files(input_dir: str | Path) -> dict[str, SourceFileRecord]:
    root = Path(input_dir)
    records: dict[str, SourceFileRecord] = {}
    if not root.exists():
        return records

    for path in sorted(root.rglob("*")):
        if not is_supported_file(path):
            continue
        relative_path = path.relative_to(root).as_posix()
        stat = path.stat()
        records[relative_path] = SourceFileRecord(
            path=relative_path,
            sha256=hash_file(path),
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            chunk_ids=[],
        )
    return records


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
