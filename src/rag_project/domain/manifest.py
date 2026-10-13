"""Index manifest: which source files an index was built from.

The manifest drives incremental indexing. Comparing it with the current set
of source files tells whether new files can simply be appended or whether
the index must be rebuilt (FAISS cannot reliably update vectors in place).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from rag_project.domain.models import Document


@dataclass(frozen=True)
class SourceFileRecord:
    path: str
    sha256: str
    size: int
    mtime_ns: int
    chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ManifestDiff:
    """Difference between an indexed manifest and the current source files."""

    added: list[str]
    changed: list[str]
    deleted: frozenset[str]

    @property
    def requires_rebuild(self) -> bool:
        return bool(self.changed or self.deleted)


@dataclass(frozen=True)
class IndexManifest:
    files: dict[str, SourceFileRecord]
    version: int = 1

    @classmethod
    def empty(cls) -> IndexManifest:
        return cls(files={})

    @classmethod
    def from_chunks(
        cls,
        records: Mapping[str, SourceFileRecord],
        chunks: Iterable[Document],
    ) -> IndexManifest:
        """Manifest for a freshly built index covering every scanned file."""
        chunk_ids = chunk_ids_by_source(chunks)
        return cls(
            files={
                path: with_chunk_ids(record, chunk_ids.get(path, []))
                for path, record in records.items()
            }
        )

    def diff(self, current: Mapping[str, SourceFileRecord]) -> ManifestDiff:
        return ManifestDiff(
            added=[path for path in current if path not in self.files],
            changed=[
                path
                for path, record in current.items()
                if path in self.files and record.sha256 != self.files[path].sha256
            ],
            deleted=frozenset(self.files) - frozenset(current),
        )

    def with_appended(
        self,
        records: Mapping[str, SourceFileRecord],
        added: Iterable[str],
        chunks: Iterable[Document],
    ) -> IndexManifest:
        """Record newly appended source files, including ones with no chunks.

        Recording chunk-less files (blank text, image-only PDFs) keeps them from
        being reloaded on every incremental run, matching a full build.
        """
        chunk_ids = chunk_ids_by_source(chunks)
        files = dict(self.files)
        for path in added:
            files[path] = with_chunk_ids(records[path], chunk_ids.get(path, []))
        return replace(self, files=files)

    def without(self, path: str) -> IndexManifest:
        files = dict(self.files)
        files.pop(path, None)
        return replace(self, files=files)


def with_chunk_ids(record: SourceFileRecord, chunk_ids: list[str]) -> SourceFileRecord:
    return replace(record, chunk_ids=chunk_ids)


def chunk_ids_by_source(chunks: Iterable[Document]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        if chunk.relative_path:
            grouped[chunk.relative_path].append(chunk.id)
    return dict(grouped)
