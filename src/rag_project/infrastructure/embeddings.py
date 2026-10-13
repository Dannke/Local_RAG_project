"""Embedding model adapters."""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Sequence
from typing import Any


class SentenceTransformerEmbeddingModel:
    """sentence-transformers embedder, loaded lazily on first use.

    Construction is cheap, so callers (e.g. an incremental ingest with nothing
    new to index) never pay for loading the model unless they embed or count
    tokens.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._lock = threading.Lock()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()

    def count_tokens(self, text: str) -> int:
        """Count tokens the way the model's own tokenizer sees ``text``."""
        if not text:
            return 0
        return len(self._get_model().tokenizer.encode(text))

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = _load_sentence_transformer(self.model_name)
        return self._model


def _load_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install sentence-transformers: python -m pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


class HashingEmbeddingModel:
    """Small deterministic embedder useful for local smoke tests."""

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")
        self.dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions

        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
