"""File manifest helpers for incremental indexing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass(frozen=True)
class SourceFileRecord:
    path: str
    sha256: str
    size: int
    mtime_ns: int
    chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IndexManifest:
    files: dict[str, SourceFileRecord]
    version: int = 1

    @classmethod
    def empty(cls) -> "IndexManifest":
        return cls(files={})

    @classmethod
    def load(cls, index_dir: str | Path) -> "IndexManifest":
        path = manifest_path(index_dir)
        if not path.exists():
            return cls.empty()
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = {
            key: SourceFileRecord(**value)
            for key, value in payload.get("files", {}).items()
        }
        return cls(files=files, version=payload.get("version", 1))

    def save(self, index_dir: str | Path) -> None:
        path = manifest_path(index_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "files": {
                key: {
                    "path": value.path,
                    "sha256": value.sha256,
                    "size": value.size,
                    "mtime_ns": value.mtime_ns,
                    "chunk_ids": value.chunk_ids,
                }
                for key, value in self.files.items()
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def manifest_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / "manifest.json"


def scan_source_files(input_dir: str | Path) -> dict[str, SourceFileRecord]:
    root = Path(input_dir)
    records: dict[str, SourceFileRecord] = {}
    if not root.exists():
        return records

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SOURCE_EXTENSIONS:
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


def with_chunk_ids(record: SourceFileRecord, chunk_ids: list[str]) -> SourceFileRecord:
    return SourceFileRecord(
        path=record.path,
        sha256=record.sha256,
        size=record.size,
        mtime_ns=record.mtime_ns,
        chunk_ids=chunk_ids,
    )
