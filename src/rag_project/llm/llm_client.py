"""OpenRouter client for RAG answer generation."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

from rag_project.config import Settings, load_settings

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
    ) -> None:
        if not api_key:
            raise MissingAPIKeyError("OPENROUTER_API_KEY is not set. Add it to .env.")

        self.model = model
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self.client = client or _build_openai_client(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        client: Any | None = None,
    ) -> OpenRouterClient:
        active_settings = settings or load_settings()
        return cls(
            api_key=active_settings.openrouter_api_key or "",
            model=active_settings.openrouter_model,
            base_url=active_settings.openrouter_base_url,
            timeout_seconds=active_settings.openrouter_timeout_seconds,
            temperature=active_settings.temperature,
            max_context_chars=active_settings.max_context_chars,
            client=client,
        )

    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        limited_chunks = limit_context_chunks(context_chunks, self.max_context_chars)
        user_prompt = build_user_prompt(question, limited_chunks)
        _log_prompt_stats(self.model, context_chunks, limited_chunks, user_prompt)

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
            raise _map_provider_error(exc) from exc

        answer = _extract_answer_text(completion)
        if not answer:
            raise EmptyLLMResponseError("OpenRouter returned an empty response.")
        return answer

    def stream_generate_answer(self, question: str, context_chunks: list[str]) -> Iterator[str]:
        limited_chunks = limit_context_chunks(context_chunks, self.max_context_chars)
        user_prompt = build_user_prompt(question, limited_chunks)
        _log_prompt_stats(self.model, context_chunks, limited_chunks, user_prompt)

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
            raise _map_provider_error(exc) from exc

        emitted = False
        try:
            for chunk in stream:
                text = _extract_stream_delta(chunk)
                if text:
                    emitted = True
                    yield text
        except Exception as exc:
            raise _map_provider_error(exc) from exc

        if not emitted:
            raise EmptyLLMResponseError("OpenRouter returned an empty streaming response.")


class OpenRouterGenerator:
    """Adapter used by the chat pipeline."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenRouterClient | None = None,
    ) -> None:
        self.settings = settings
        self.client = client

    def generate(self, question: str, context_chunks: Sequence[str]) -> str:
        client = self.client or OpenRouterClient.from_settings(self.settings)
        return client.generate_answer(question, list(context_chunks))

    def stream(self, question: str, context_chunks: Sequence[str]) -> Iterator[str]:
        client = self.client or OpenRouterClient.from_settings(self.settings)
        yield from client.stream_generate_answer(question, list(context_chunks))


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
    logger.info("Using OpenRouter model: %s", model)
    logger.info("Retrieved chunks found: %s", len(context_chunks))
    logger.info("Chunks sent after context limit: %s", len(limited_chunks))
    logger.info("Prompt length: %s chars", len(SYSTEM_PROMPT) + len(user_prompt))


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
