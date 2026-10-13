"""On-disk layout of one FAISS index directory.

Owns the knowledge of which files make up an index (vectors, document
metadata, manifest) so neither use cases nor the UI depend on file names.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_project.domain.manifest import IndexManifest, SourceFileRecord
from rag_project.infrastructure.faiss_store import (
    DOCUMENTS_FILENAME,
    INDEX_FILENAME,
    FaissVectorStore,
    read_document_count,
)
from rag_project.infrastructure.filesystem import safe_unlink

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
INDEX_FILES = (INDEX_FILENAME, DOCUMENTS_FILENAME, MANIFEST_FILENAME)


@dataclass(frozen=True)
class IndexStats:
    chunk_count: int | None
    size_bytes: int


class FaissIndexRepository:
    """Implements :class:`~rag_project.application.ports.IndexRepository`."""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)

    def exists(self) -> bool:
        return (self.index_dir / INDEX_FILENAME).exists() and (
            self.index_dir / DOCUMENTS_FILENAME
        ).exists()

    def new_store(self) -> FaissVectorStore:
        return FaissVectorStore(self.index_dir)

    def load_store(self) -> FaissVectorStore:
        return FaissVectorStore.load_from_disk(self.index_dir)

    def load_manifest(self) -> IndexManifest:
        """Load the manifest; a missing or corrupt one reads as empty.

        An empty manifest makes the next incremental ingest rebuild the index,
        so a damaged file heals itself instead of failing every run.
        """
        path = self.index_dir / MANIFEST_FILENAME
        if not path.exists():
            return IndexManifest.empty()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            files = {
                key: SourceFileRecord(**value) for key, value in payload.get("files", {}).items()
            }
            return IndexManifest(files=files, version=payload.get("version", 1))
        except (ValueError, TypeError, AttributeError) as exc:
            logger.warning(
                "manifest_unreadable path=%s error=%s",
                path,
                exc,
                extra={"path": str(path), "error": str(exc)},
                exc_info=True,
            )
            return IndexManifest.empty()

    def save_manifest(self, manifest: IndexManifest) -> None:
        path = self.index_dir / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": manifest.version,
            "files": {key: asdict(record) for key, record in manifest.files.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> int:
        """Delete the index files; return how many were removed."""
        removed = 0
        for path in self._existing_files():
            safe_unlink(path, self.index_dir)
            removed += 1
        return removed

    def stats(self) -> IndexStats:
        return IndexStats(
            chunk_count=read_document_count(self.index_dir / DOCUMENTS_FILENAME),
            size_bytes=sum(path.stat().st_size for path in self._existing_files()),
        )

    def _existing_files(self) -> list[Path]:
        if not self.index_dir.exists():
            return []
        return [
            self.index_dir / name for name in INDEX_FILES if (self.index_dir / name).exists()
        ]
