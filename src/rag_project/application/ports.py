"""Ports: the interfaces use cases need from the outside world.

Adapters in ``rag_project.infrastructure`` satisfy them structurally; tests
satisfy them with in-memory fakes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from rag_project.domain.manifest import IndexManifest, SourceFileRecord
from rag_project.domain.models import Document, SearchResult


class EmbeddingModel(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per text."""
        ...


class VectorStore(Protocol):
    def add(self, documents: Sequence[Document], embeddings: Sequence[Sequence[float]]) -> None:
        """Add documents and their embeddings to the store."""
        ...

    def search(self, embedding: Sequence[float], top_k: int = 5) -> list[SearchResult]:
        """Return the nearest documents for an embedding."""
        ...

    def count(self) -> int:
        """Return the number of stored documents."""
        ...

    def remove_ids(self, document_ids: Sequence[str]) -> int:
        """Remove documents by their IDs; return the number removed."""
        ...


class PersistentVectorStore(VectorStore, Protocol):
    def save(self) -> None:
        """Persist the store to its backing storage."""
        ...

    def ids_for_source(self, relative_path: str) -> list[str]:
        """IDs of the stored chunks that came from one source file."""
        ...


class Reranker(Protocol):
    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        """Return the ``top_k`` most relevant results, best first."""
        ...


class AnswerGenerator(Protocol):
    """Generates an answer from a question and formatted context chunks.

    Implementations may raise :class:`~rag_project.domain.errors.LLMClientError`.
    """

    def generate(self, question: str, context_chunks: Sequence[str]) -> str: ...

    def stream(self, question: str, context_chunks: Sequence[str]) -> Iterator[str]: ...


class DocumentSource(Protocol):
    """A collection of source files addressed by relative path."""

    def scan(self) -> dict[str, SourceFileRecord]:
        """Fingerprint every supported source file."""
        ...

    def load_all(self) -> list[Document]:
        """Load every supported source file."""
        ...

    def load(self, relative_path: str) -> list[Document]:
        """Load a single source file."""
        ...


class IndexRepository(Protocol):
    """Persistent storage for one vector index and its manifest."""

    def exists(self) -> bool: ...

    def new_store(self) -> PersistentVectorStore: ...

    def load_store(self) -> PersistentVectorStore:
        """Load the saved store; raises ``FileNotFoundError`` if there is none."""
        ...

    def load_manifest(self) -> IndexManifest:
        """Saved manifest; a missing or unreadable one reads as empty (never raises).

        An empty manifest makes incremental indexing rebuild the index.
        """
        ...

    def save_manifest(self, manifest: IndexManifest) -> None: ...
