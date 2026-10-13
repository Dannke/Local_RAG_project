"""Domain-level error types raised through the application ports."""

from __future__ import annotations


class NoDocumentsToIndexError(RuntimeError):
    """Raised when a full index build finds no indexable documents."""


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
