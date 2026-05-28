"""FAISS-backed vector store with JSON document metadata."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rag_project.models import Document, SearchResult


class FaissVectorStore:
    def __init__(self, index_dir: str | Path) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install FAISS: python -m pip install faiss-cpu") from exc

        self._faiss = faiss
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index.faiss"
        self.documents_path = self.index_dir / "documents.json"
        self.index: Any | None = None
        self.documents: list[Document] = []

    @classmethod
    def load_from_disk(cls, index_dir: str | Path) -> "FaissVectorStore":
        store = cls(index_dir)
        store.load()
        return store

    def add(self, documents: Sequence[Document], embeddings: Sequence[Sequence[float]]) -> None:
        vectors = _to_float32_matrix(embeddings)
        if len(documents) != len(vectors):
            raise ValueError("documents and embeddings must have the same length")
        if len(vectors) == 0:
            return

        if self.index is None:
            self.index = self._faiss.IndexFlatIP(vectors.shape[1])

        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, embedding: Sequence[float], top_k: int = 5) -> list[SearchResult]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query = _to_float32_matrix([embedding])
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        results: list[SearchResult] = []

        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            results.append(SearchResult(document=self.documents[int(index)], score=float(score)))

        return results

    def count(self) -> int:
        return len(self.documents)

    def save(self) -> None:
        self.save_to_disk()

    def save_to_disk(self) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an empty FAISS index")
        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                f"FAISS index contains {self.index.ntotal} vectors, "
                f"but metadata contains {len(self.documents)} documents"
            )

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(self.index_path))
        payload = [asdict(document) for document in self.documents]
        self.documents_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        self.load_from_disk_into_self()

    def load_from_disk_into_self(self) -> None:
        if not self.index_path.exists() or not self.documents_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found in {self.index_dir}. Run ingest before search."
            )

        self.index = self._faiss.read_index(str(self.index_path))
        payload = json.loads(self.documents_path.read_text(encoding="utf-8"))
        self.documents = [
            Document(
                id=item["id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
            )
            for item in payload
        ]
        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                f"Loaded FAISS index contains {self.index.ntotal} vectors, "
                f"but metadata contains {len(self.documents)} documents"
            )


def _to_float32_matrix(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    return matrix
