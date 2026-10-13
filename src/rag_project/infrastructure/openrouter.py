"""OpenRouter (OpenAI-compatible) adapter for RAG answer generation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from rag_project.application.observability import now_ms
from rag_project.domain.errors import (
    EmptyLLMResponseError,
    InvalidModelError,
    LLMClientError,
    LLMProviderError,
    LLMTimeoutError,
    MissingAPIKeyError,
)
from rag_project.domain.prompts import build_messages, limit_context_chunks
from rag_project.infrastructure.rate_limit import RateLimiter, RateLimitExceededError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: int = 60,
        temperature: float = 0.2,
        max_context_chars: int = 12_000,
        client: Any | None = None,
        rate_limiter: RateLimiter | None = None,
        rate_limit_key: str | None = None,
    ) -> None:
        if not api_key:
            raise MissingAPIKeyError("OPENROUTER_API_KEY is not set. Add it to .env.")

        self.model = model
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self.rate_limiter = rate_limiter
        self.rate_limit_key = rate_limit_key
        self.client = client or _build_openai_client(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    def generate_answer(self, question: str, context_chunks: Sequence[str]) -> str:
        messages = self._prepare(question, context_chunks)
        start = now_ms()
        try:
            completion = self._create(messages)
        except Exception as exc:
            raise self._provider_error(exc, start) from exc

        answer = _extract_answer_text(completion)
        if not answer:
            raise EmptyLLMResponseError("OpenRouter returned an empty response.")

        logger.info(
            "llm_completed",
            extra={
                "model": self.model,
                "latency_ms": now_ms() - start,
                "success": True,
                **_extract_usage(completion),
            },
        )
        return answer

    def stream_generate_answer(
        self, question: str, context_chunks: Sequence[str]
    ) -> Iterator[str]:
        messages = self._prepare(question, context_chunks)
        start = now_ms()
        emitted = False
        usage: dict[str, int] = {}
        try:
            for chunk in self._create(messages, stream=True):
                text = _extract_stream_delta(chunk)
                if text:
                    emitted = True
                    yield text
                if getattr(chunk, "usage", None) is not None:
                    usage = _extract_usage(chunk)
        except Exception as exc:
            raise self._provider_error(exc, start) from exc

        logger.info(
            "llm_stream_completed",
            extra={
                "model": self.model,
                "latency_ms": now_ms() - start,
                "success": emitted,
                "empty": not emitted,
                **usage,
            },
        )
        if not emitted:
            raise EmptyLLMResponseError("OpenRouter returned an empty streaming response.")

    def _prepare(self, question: str, context_chunks: Sequence[str]) -> list[dict[str, str]]:
        self._check_rate_limit()
        limited_chunks = limit_context_chunks(context_chunks, self.max_context_chars)
        messages = build_messages(question, limited_chunks)
        logger.info(
            "llm_prompt",
            extra={
                "model": self.model,
                "chunks_retrieved": len(context_chunks),
                "chunks_after_limit": len(limited_chunks),
                "user_prompt_chars": len(messages[-1]["content"]),
            },
        )
        return messages

    def _create(self, messages: list[dict[str, str]], stream: bool = False) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if stream:
            kwargs["stream"] = True
        return self.client.chat.completions.create(**kwargs)

    def _check_rate_limit(self) -> None:
        if self.rate_limiter is None:
            return
        try:
            self.rate_limiter.check(self.rate_limit_key or "default")
        except RateLimitExceededError as exc:
            logger.warning("LLM request rejected by rate limiter: %s", exc)
            raise

    def _provider_error(self, exc: Exception, start: int) -> LLMClientError:
        mapped = _map_provider_error(exc)
        logger.warning(
            "llm_error",
            extra={
                "model": self.model,
                "latency_ms": now_ms() - start,
                "error": mapped.__class__.__name__,
                "provider_error": exc.__class__.__name__,
            },
        )
        return mapped


class OpenRouterGenerator:
    """:class:`~rag_project.application.ports.AnswerGenerator` over OpenRouter.

    The client is created on first use and then cached, so a missing API key
    surfaces when a question is asked (not when a chat session is opened) and
    the per-client rate limiter is shared by all calls of this generator.
    """

    def __init__(self, client_factory: Callable[[], OpenRouterClient]) -> None:
        self._client_factory = client_factory
        self._client: OpenRouterClient | None = None

    def generate(self, question: str, context_chunks: Sequence[str]) -> str:
        return self._get_client().generate_answer(question, context_chunks)

    def stream(self, question: str, context_chunks: Sequence[str]) -> Iterator[str]:
        yield from self._get_client().stream_generate_answer(question, context_chunks)

    def _get_client(self) -> OpenRouterClient:
        if self._client is None:
            self._client = self._client_factory()
        return self._client


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from a completion or a final streamed chunk."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    try:
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
    except (AttributeError, TypeError, ValueError):
        return {}


def _build_openai_client(api_key: str, base_url: str, timeout_seconds: int) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMClientError(
            "Install openai to use OpenRouter: python -m pip install openai"
        ) from exc

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
    )


def _extract_answer_text(completion: Any) -> str:
    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise EmptyLLMResponseError(
            "OpenRouter response does not contain message content."
        ) from exc

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [getattr(part, "text", "") for part in content]
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _extract_stream_delta(chunk: Any) -> str:
    try:
        content = chunk.choices[0].delta.content
    except (AttributeError, IndexError, TypeError):
        return ""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [getattr(part, "text", "") for part in content]
        return "".join(parts)
    return ""


def _map_provider_error(exc: Exception) -> LLMClientError:
    if isinstance(exc, LLMClientError):
        return exc

    exc_name = exc.__class__.__name__
    message = str(exc)

    if exc_name == "APITimeoutError":
        return LLMTimeoutError(f"OpenRouter request timed out: {message}")
    if exc_name in {"BadRequestError", "NotFoundError"} and "model" in message.lower():
        return InvalidModelError(f"OpenRouter model is invalid or unavailable: {message}")
    return LLMProviderError(f"OpenRouter request failed: {message}")
