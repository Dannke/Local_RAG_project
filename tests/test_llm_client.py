import unittest
from types import SimpleNamespace

from rag_project.llm.llm_client import (
    SYSTEM_PROMPT,
    EmptyLLMResponseError,
    MissingAPIKeyError,
    OpenRouterClient,
    build_user_prompt,
    limit_context_chunks,
)


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        if kwargs.get("stream"):
            return [
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Final "))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="answer."))]),
            ]
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeOpenAIClient:
    def __init__(self, content="Final answer from OpenRouter."):
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


class LLMClientTests(unittest.TestCase):
    def test_openrouter_client_sends_context_and_system_prompt(self) -> None:
        fake_client = FakeOpenAIClient()
        client = OpenRouterClient(
            api_key="test-key",
            model="test/model",
            client=fake_client,
            temperature=0.3,
            max_context_chars=100,
        )

        answer = client.generate_answer("What is indexed?", ["FAISS chunks."])

        self.assertEqual(answer, "Final answer from OpenRouter.")
        self.assertEqual(fake_client.completions.request["model"], "test/model")
        self.assertEqual(fake_client.completions.request["temperature"], 0.3)
        self.assertEqual(fake_client.completions.request["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertIn("FAISS chunks.", fake_client.completions.request["messages"][1]["content"])

    def test_openrouter_client_streams_answer(self) -> None:
        fake_client = FakeOpenAIClient()
        client = OpenRouterClient(
            api_key="test-key",
            model="test/model",
            client=fake_client,
        )

        answer = "".join(client.stream_generate_answer("What is indexed?", ["FAISS chunks."]))

        self.assertEqual(answer, "Final answer.")
        self.assertTrue(fake_client.completions.request["stream"])

    def test_missing_api_key_raises_clear_error(self) -> None:
        with self.assertRaises(MissingAPIKeyError):
            OpenRouterClient(api_key="", model="test/model", client=FakeOpenAIClient())

    def test_empty_response_raises_clear_error(self) -> None:
        client = OpenRouterClient(api_key="test-key", model="test/model", client=FakeOpenAIClient(""))

        with self.assertRaises(EmptyLLMResponseError):
            client.generate_answer("Question?", ["Context."])

    def test_context_is_limited_before_prompt(self) -> None:
        chunks = limit_context_chunks(["abc", "defgh", "ijk"], max_chars=6)
        prompt = build_user_prompt("Question?", chunks)

        self.assertEqual(chunks, ["abc", "def"])
        self.assertIn("Контекст:", prompt)
        self.assertIn("Вопрос:", prompt)
