import unittest

from rag_project.llm.llm_client import OpenRouterClient
from rag_project.rate_limit import RateLimiter, RateLimitExceededError

try:
    from types import SimpleNamespace
except ImportError:  # pragma: no cover
    SimpleNamespace = None


class FakeCompletions:
    def __init__(self, content="ok"):
        self.content = content

    def create(self, **kwargs):
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class RateLimiterTests(unittest.TestCase):
    def test_allows_requests_up_to_max(self) -> None:
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        self.assertTrue(limiter.allow("session-a"))
        self.assertTrue(limiter.allow("session-a"))
        self.assertTrue(limiter.allow("session-a"))
        self.assertFalse(limiter.allow("session-a"))

    def test_keys_are_independent(self) -> None:
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        self.assertTrue(limiter.allow("a"))
        self.assertTrue(limiter.allow("a"))
        self.assertFalse(limiter.allow("a"))
        self.assertTrue(limiter.allow("b"))

    def test_check_raises_when_exceeded(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.check("session")
        with self.assertRaises(RateLimitExceededError):
            limiter.check("session")

    def test_retry_after_is_positive(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=30)
        limiter.check("session")
        with self.assertRaises(RateLimitExceededError) as ctx:
            limiter.check("session")
        self.assertGreater(ctx.exception.retry_after, 0)

    def test_reset_clears_history(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        self.assertTrue(limiter.allow("a"))
        limiter.reset("a")
        self.assertTrue(limiter.allow("a"))


class RateLimiterLLMIntegrationTests(unittest.TestCase):
    def test_client_rejects_when_over_limit(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        client = OpenRouterClient(
            api_key="test-key",
            model="test/model",
            client=FakeOpenAIClient(),
            rate_limiter=limiter,
            rate_limit_key="chat-1",
        )
        client.generate_answer("Q", ["ctx"])
        with self.assertRaises(RateLimitExceededError):
            client.generate_answer("Q", ["ctx"])

    def test_disabled_when_no_limiter(self) -> None:
        client = OpenRouterClient(
            api_key="test-key",
            model="test/model",
            client=FakeOpenAIClient(),
        )
        for _ in range(5):
            self.assertEqual(client.generate_answer("Q", ["ctx"]), "ok")


if __name__ == "__main__":
    unittest.main()
