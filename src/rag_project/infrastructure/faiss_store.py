"""FAISS-backed vector store with JSON document metadata.

Uses :class:`IndexIDMap` on top of ``IndexFlatIP`` so that each vector has a
stable integer id. This enables ``remove_ids`` (document deletion) which plain
``IndexFlatIP`` does not support.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rag_project.domain.models import Document, SearchResult
from rag_project.infrastructure.serialization import document_from_dict, document_to_dict

INDEX_FILENAME = "index.faiss"
DOCUMENTS_FILENAME = "documents.json"


class FaissVectorStore:
    def __init__(self, index_dir: str | Path) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install FAISS: python -m pip install faiss-cpu") from exc

        self._faiss = faiss
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / INDEX_FILENAME
        self.documents_path = self.index_dir / DOCUMENTS_FILENAME
        self.index: Any | None = None
        self.documents: list[Document] = []
        # Parallel to `self.documents`: the integer id FAISS uses for each doc.
        self._faiss_ids: list[int] = []
        # Maps integer FAISS id -> position in self.documents.
        self._id_to_position: dict[int, int] = {}

    @classmethod
    def load_from_disk(cls, index_dir: str | Path) -> FaissVectorStore:
        store = cls(index_dir)
        store.load()
        return store

    def add(self, documents: Sequence[Document], embeddings: Sequence[Sequence[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")
        if not documents:
            return

        vectors = _to_float32_matrix(embeddings)
        if self.index is None:
            base = self._faiss.IndexFlatIP(vectors.shape[1])
            self.index = self._faiss.IndexIDMap(base)

        start = self._next_faiss_id()
        new_ids = list(range(start, start + len(vectors)))
        self.index.add_with_ids(vectors, np.asarray(new_ids, dtype="int64"))
        first_position = len(self.documents)
        self.documents.extend(documents)
        self._faiss_ids.extend(new_ids)
        for position, faiss_id in enumerate(new_ids, start=first_position):
            self._id_to_position[faiss_id] = position

    def remove_ids(self, document_ids: Sequence[str]) -> int:
        """Remove documents by their string ``Document.id``.

        Returns the number of documents removed. Vectors are deleted from the
        FAISS index and the metadata list is rebuilt; other documents are
        untouched.
        """
        if self.index is None or not self.documents:
            return 0

        remove_set = set(document_ids)
        remove_positions = {
            position
            for position, document in enumerate(self.documents)
            if document.id in remove_set
        }
        if not remove_positions:
            return 0

        self.index.remove_ids(
            np.asarray(
                [self._faiss_ids[position] for position in sorted(remove_positions)],
                dtype="int64",
            )
        )
        keep = [
            position for position in range(len(self.documents)) if position not in remove_positions
        ]
        self.documents = [self.documents[position] for position in keep]
        self._faiss_ids = [self._faiss_ids[position] for position in keep]
        self._rebuild_id_mapping()
        return len(remove_positions)

    def search(self, embedding: Sequence[float], top_k: int = 5) -> list[SearchResult]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query = _to_float32_matrix([embedding])
        scores, ids = self.index.search(query, min(top_k, self.index.ntotal))
        results: list[SearchResult] = []

        for score, faiss_id in zip(scores[0], ids[0], strict=True):
            if faiss_id < 0:
                continue
            position = self._id_to_position.get(int(faiss_id))
            if position is None:
                continue
            results.append(SearchResult(document=self.documents[position], score=float(score)))

        return results

    def count(self) -> int:
        return len(self.documents)

    def ids_for_source(self, relative_path: str) -> list[str]:
        return [
            document.id for document in self.documents if document.relative_path == relative_path
        ]

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an empty FAISS index")
        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                f"FAISS index contains {self.index.ntotal} vectors, "
                f"but metadata contains {len(self.documents)} documents"
            )

        payload = {
            "documents": [document_to_dict(document) for document in self.documents],
            "faiss_ids": self._faiss_ids,
        }
        documents_json = json.dumps(payload, ensure_ascii=False, indent=2)

        # Write both files aside first so a failure mid-save (disk full,
        # unserializable metadata) leaves the previously saved index intact.
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_tmp = self.index_path.with_name(self.index_path.name + ".tmp")
        documents_tmp = self.documents_path.with_name(self.documents_path.name + ".tmp")
        self._faiss.write_index(self.index, str(index_tmp))
        documents_tmp.write_text(documents_json, encoding="utf-8")
        os.replace(index_tmp, self.index_path)
        os.replace(documents_tmp, self.documents_path)

    def load(self) -> None:
        if not self.index_path.exists() or not self.documents_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found in {self.index_dir}. Run ingest before search."
            )

        self.index = self._faiss.read_index(str(self.index_path))
        items, faiss_ids = _parse_documents_payload(
            json.loads(self.documents_path.read_text(encoding="utf-8"))
        )
        self.documents = [document_from_dict(item) for item in items]
        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                f"Loaded FAISS index contains {self.index.ntotal} vectors, "
                f"but metadata contains {len(self.documents)} documents"
            )

        self._faiss_ids = [int(faiss_id) for faiss_id in faiss_ids]
        self._rebuild_id_mapping()

    def _next_faiss_id(self) -> int:
        if not self._faiss_ids:
            return 0
        return max(self._faiss_ids) + 1

    def _rebuild_id_mapping(self) -> None:
        self._id_to_position = {
            faiss_id: position for position, faiss_id in enumerate(self._faiss_ids)
        }


def read_document_count(documents_path: Path) -> int | None:
    """Number of documents stored in ``documents.json`` without loading FAISS.

    Returns ``None`` when the file is missing or unreadable.
    """
    try:
        payload = json.loads(documents_path.read_text(encoding="utf-8"))
        items, _ = _parse_documents_payload(payload)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return len(items)


def _parse_documents_payload(payload: Any) -> tuple[list[dict[str, Any]], list[int]]:
    # Current on-disk format: {"documents": [...], "faiss_ids": [...]}.
    # The original format was a bare list of document dicts; fall back to
    # sequential ids for backward compatibility.
    if isinstance(payload, dict):
        items = payload["documents"]
        return items, payload.get("faiss_ids") or list(range(len(items)))
    return payload, list(range(len(payload)))


def _to_float32_matrix(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    return matrix
