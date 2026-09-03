"""Sentence-transformers embedding provider."""

from __future__ import annotations

from collections.abc import Callable, Sequence


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers: python -m pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()

    def token_counter(self) -> Callable[[str], int]:
        """Return a callable that counts tokens the way the model sees text.

        Uses the tokenizer of the same tokenizer the model was built with.
        """
        def count(text: str) -> int:
            if not text:
                return 0
            return len(self._model.tokenizer.encode(text))

        return count
