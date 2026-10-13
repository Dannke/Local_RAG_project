import threading
import time
import unittest
from unittest import mock

import numpy as np

from rag_project.domain.errors import MissingAPIKeyError
from rag_project.infrastructure import embeddings
from rag_project.infrastructure.embeddings import SentenceTransformerEmbeddingModel
from rag_project.infrastructure.openrouter import OpenRouterGenerator

MODEL_NAME = "fake/embedding-model"


class FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.encode_calls: list[tuple[list[str], dict]] = []

    def encode(self, texts, **kwargs) -> np.ndarray:
        self.encode_calls.append((list(texts), kwargs))
        return np.asarray([[float(len(text)), 1.0, 0.5] for text in texts], dtype="float64")


class CountingLoader:
    """Stands in for ``_load_sentence_transformer``; never touches the network."""

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.model = FakeSentenceTransformer()
        self.model_names: list[str] = []
        self._delay_seconds = delay_seconds

    def __call__(self, model_name: str) -> FakeSentenceTransformer:
        self.model_names.append(model_name)
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        return self.model


class SentenceTransformerLazyLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = CountingLoader()
        patcher = mock.patch.object(embeddings, "_load_sentence_transformer", self.loader)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_construction_does_not_load_the_model(self) -> None:
        model = SentenceTransformerEmbeddingModel(MODEL_NAME)

        self.assertEqual(model.model_name, MODEL_NAME)
        self.assertEqual(self.loader.model_names, [])

    def test_model_is_loaded_once_across_embeds_and_token_counts(self) -> None:
        model = SentenceTransformerEmbeddingModel(MODEL_NAME)

        first = model.embed_texts(["a", "bb"])
        second = model.embed_texts(["ccc"])
        tokens = model.count_tokens("one two three")

        self.assertEqual(self.loader.model_names, [MODEL_NAME])
        self.assertEqual(first, [[1.0, 1.0, 0.5], [2.0, 1.0, 0.5]])
        self.assertEqual(second, [[3.0, 1.0, 0.5]])
        self.assertIsInstance(first[0][0], float)
        self.assertEqual(tokens, 3)

    def test_embeddings_are_requested_normalized_without_progress_bar(self) -> None:
        SentenceTransformerEmbeddingModel(MODEL_NAME).embed_texts(("x", "y"))

        texts, kwargs = self.loader.model.encode_calls[0]
        self.assertEqual(texts, ["x", "y"])
        self.assertTrue(kwargs["normalize_embeddings"])
        self.assertTrue(kwargs["convert_to_numpy"])
        self.assertFalse(kwargs["show_progress_bar"])

    def test_counting_tokens_of_empty_text_does_not_load_the_model(self) -> None:
        model = SentenceTransformerEmbeddingModel(MODEL_NAME)

        self.assertEqual(model.count_tokens(""), 0)
        self.assertEqual(self.loader.model_names, [])

    def test_concurrent_first_use_loads_the_model_once(self) -> None:
        slow_loader = CountingLoader(delay_seconds=0.05)
        model = SentenceTransformerEmbeddingModel(MODEL_NAME)
        barrier = threading.Barrier(8)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=5)
                model.embed_texts(["text"])
            except Exception as exc:  # surfaced through `errors` below
                errors.append(exc)

        with mock.patch.object(embeddings, "_load_sentence_transformer", slow_loader):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(slow_loader.model_names, [MODEL_NAME])
        self.assertEqual(len(slow_loader.model.encode_calls), 8)


class FakeOpenRouterClient:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def generate_answer(self, question: str, context_chunks) -> str:
        self.questions.append(question)
        return f"answer to {question} from {len(context_chunks)} chunks"

    def stream_generate_answer(self, question: str, context_chunks):
        self.questions.append(question)
        yield "streamed "
        yield "answer"


class CountingFactory:
    def __init__(self, error: Exception | None = None) -> None:
        self.client = FakeOpenRouterClient()
        self.calls = 0
        self._error = error

    def __call__(self) -> FakeOpenRouterClient:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self.client


class OpenRouterGeneratorLazyClientTests(unittest.TestCase):
    def test_construction_does_not_create_the_client(self) -> None:
        factory = CountingFactory()

        OpenRouterGenerator(factory)

        self.assertEqual(factory.calls, 0)

    def test_client_is_created_once_and_reused(self) -> None:
        factory = CountingFactory()
        generator = OpenRouterGenerator(factory)

        first = generator.generate("q1", ["chunk"])
        streamed = "".join(generator.stream("q2", []))
        third = generator.generate("q3", ["a", "b"])

        self.assertEqual(factory.calls, 1)
        self.assertEqual(first, "answer to q1 from 1 chunks")
        self.assertEqual(streamed, "streamed answer")
        self.assertEqual(third, "answer to q3 from 2 chunks")
        self.assertEqual(factory.client.questions, ["q1", "q2", "q3"])

    def test_stream_creates_the_client_only_when_iterated(self) -> None:
        factory = CountingFactory()
        generator = OpenRouterGenerator(factory)

        stream = generator.stream("q", [])
        self.assertEqual(factory.calls, 0)

        self.assertEqual(list(stream), ["streamed ", "answer"])
        self.assertEqual(factory.calls, 1)

    def test_factory_errors_surface_on_first_use_not_on_construction(self) -> None:
        factory = CountingFactory(error=MissingAPIKeyError("OPENROUTER_API_KEY is not set."))
        generator = OpenRouterGenerator(factory)

        with self.assertRaises(MissingAPIKeyError):
            generator.generate("q", [])

        self.assertEqual(factory.calls, 1)
