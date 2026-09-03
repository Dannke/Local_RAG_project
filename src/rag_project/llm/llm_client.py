"""OpenRouter client for RAG answer generation."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from typing import Any

from rag_project.config import Settings, load_settings
from rag_project.rate_limit import RateLimiter, RateLimitExceededError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты AI assistant для RAG-системы.\n"
    "Отвечай ТОЛЬКО на основе предоставленного контекста.\n"
    "Если информации недостаточно — так и скажи.\n"
    "Не придумывай факты.\n"
    "Если ответ найден, указывай источники ссылками вида [S1], [S2]."
)


class LLMClientError(RuntimeError):
    """Base error for LLM provider failures."""


class MissingAPIKeyError(LLMClientError):
    """Raised when OPENROUTER_API_KEY is missing."""


class LLMTimeoutError(LLMClientError):
    """Raised when the LLM request times out."""


class InvalidModelError(LLMClientError):
    """Raised when OpenRouter rejects the selected model."""


class EmptyLLMResponseError(LLMClientError):
    """Raised when the provider returns no answer text."""


class LLMProviderError(LLMClientError):
    """Raised for other provider-side failures."""


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
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

    def _check_rate_limit(self) -> None:
        if self.rate_limiter is None:
            return
        try:
            self.rate_limiter.check(self.rate_limit_key or "default")
        except RateLimitExceededError as exc:
            logger.warning("LLM request rejected by rate limiter: %s", exc)
            raise

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        client: Any | None = None,
        rate_limit_key: str | None = None,
    ) -> OpenRouterClient:
        active_settings = settings or load_settings()
        rate_limiter = None
        if active_settings.rate_limit_max_requests > 0:
            rate_limiter = RateLimiter(
                max_requests=active_settings.rate_limit_max_requests,
                window_seconds=active_settings.rate_limit_window_seconds,
            )
        return cls(
            api_key=active_settings.openrouter_api_key or "",
            model=active_settings.openrouter_model,
            base_url=active_settings.openrouter_base_url,
            timeout_seconds=active_settings.openrouter_timeout_seconds,
            temperature=active_settings.temperature,
            max_context_chars=active_settings.max_context_chars,
            client=client,
            rate_limiter=rate_limiter,
            rate_limit_key=rate_limit_key,
        )

    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        self._check_rate_limit()
        limited_chunks = limit_context_chunks(context_chunks, self.max_context_chars)
        user_prompt = build_user_prompt(question, limited_chunks)
        _log_prompt_stats(self.model, context_chunks, limited_chunks, user_prompt)

        start = _now_ms()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
        except Exception as exc:
            mapped = _map_provider_error(exc)
            logger.warning(
                "llm_error",
                extra={
                    "model": self.model,
                    "latency_ms": _now_ms() - start,
                    "error": mapped.__class__.__name__,
                },
            )
            raise mapped from exc

        answer = _extract_answer_text(completion)
        if not answer:
            raise EmptyLLMResponseError("OpenRouter returned an empty response.")

        logger.info(
            "llm_completed",
            extra={
                "model": self.model,
                "latency_ms": _now_ms() - start,
                "success": True,
                **(_extract_usage(completion)),
            },
        )
        return answer

    def stream_generate_answer(self, question: str, context_chunks: list[str]) -> Iterator[str]:
        self._check_rate_limit()
        limited_chunks = limit_context_chunks(context_chunks, self.max_context_chars)
        user_prompt = build_user_prompt(question, limited_chunks)
        _log_prompt_stats(self.model, context_chunks, limited_chunks, user_prompt)

        start = _now_ms()
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                stream=True,
            )
        except Exception as exc:
            mapped = _map_provider_error(exc)
            logger.warning(
                "llm_error",
                extra={
                    "model": self.model,
                    "latency_ms": _now_ms() - start,
                    "error": mapped.__class__.__name__,
                },
            )
            raise mapped from exc

        emitted = False
        usage: dict = {}
        try:
            for chunk in stream:
                text = _extract_stream_delta(chunk)
                if text:
                    emitted = True
                    yield text
                if getattr(chunk, "usage", None) is not None:
                    usage = _extract_stream_usage(chunk)
        except Exception as exc:
            mapped = _map_provider_error(exc)
            logger.warning(
                "llm_error",
                extra={
                    "model": self.model,
                    "latency_ms": _now_ms() - start,
                    "error": mapped.__class__.__name__,
                },
            )
            raise mapped from exc

        logger.info(
            "llm_stream_completed",
            extra={
                "model": self.model,
                "latency_ms": _now_ms() - start,
                "success": emitted,
                "empty": not emitted,
                **usage,
            },
        )
        if not emitted:
            raise EmptyLLMResponseError("OpenRouter returned an empty streaming response.")


class OpenRouterGenerator:
    """Adapter used by the chat pipeline.

    The underlying :class:`OpenRouterClient` is created lazily and cached so
    that a rate limiter (per session) is shared across all calls of the same
    generator instance.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenRouterClient | None = None,
        rate_limit_key: str | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self.rate_limit_key = rate_limit_key

    def _get_client(self) -> OpenRouterClient:
        if self._client is None:
            self._client = OpenRouterClient.from_settings(
                self.settings, rate_limit_key=self.rate_limit_key
            )
        return self._client

    def generate(self, question: str, context_chunks: Sequence[str]) -> str:
        return self._get_client().generate_answer(question, list(context_chunks))

    def stream(self, question: str, context_chunks: Sequence[str]) -> Iterator[str]:
        yield from self._get_client().stream_generate_answer(question, list(context_chunks))


def generate_answer(question: str, context_chunks: list[str]) -> str:
    return OpenRouterClient.from_settings().generate_answer(question, context_chunks)


def stream_generate_answer(question: str, context_chunks: list[str]) -> Iterator[str]:
    yield from OpenRouterClient.from_settings().stream_generate_answer(question, context_chunks)


def build_user_prompt(question: str, context_chunks: Sequence[str]) -> str:
    chunks = "\n\n".join(
        f"[{index}]\n{chunk}" for index, chunk in enumerate(context_chunks, start=1)
    )
    return (
        "Контекст:\n"
        f"{chunks}\n\n"
        "Вопрос:\n"
        f"{question}\n\n"
        "Ответь с точными ссылками на источники в формате [S1], [S2], если ответ найден."
    )


def limit_context_chunks(context_chunks: Sequence[str], max_chars: int) -> list[str]:
    if max_chars <= 0:
        return []

    limited: list[str] = []
    used = 0
    for chunk in context_chunks:
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = chunk.strip()
        if not text:
            continue
        if len(text) > remaining:
            text = text[:remaining].rstrip()
        limited.append(text)
        used += len(text)

    return limited


def _log_prompt_stats(
    model: str,
    context_chunks: Sequence[str],
    limited_chunks: Sequence[str],
    user_prompt: str,
) -> None:
    logger.info(
        "llm_prompt",
        extra={
            "model": model,
            "chunks_retrieved": len(context_chunks),
            "chunks_after_limit": len(limited_chunks),
            "user_prompt_chars": len(user_prompt),
        },
    )


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _extract_usage(completion: Any) -> dict:
    """Extract token usage from a (non-stream) completion if present."""
    usage = getattr(completion, "usage", None)
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


def _extract_stream_usage(chunk: Any) -> dict:
    """Extract token usage from a final streamed chunk if present."""
    usage = getattr(chunk, "usage", None)
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
