"""In-memory sliding-window rate limiter.

Used to protect the LLM provider (OpenRouter) from bursts of requests,
whether caused by a UI bug or an accidental script.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Hashable


class RateLimitExceededError(Exception):
    """Raised when a request exceeds the allowed rate limit."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """Sliding-window rate limiter keyed by an arbitrary session identifier.

    Each key keeps the timestamps of recent requests. A request is allowed
    only if the number of requests within the last ``window_seconds`` is
    below ``max_requests``. Thread-safe.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self.max_requests = max_requests
        self.window_seconds = float(window_seconds)
        self._windows: dict[Hashable, deque[float]] = {}
        self._lock = threading.Lock()

    def _requests_in_window(self, key: Hashable, now: float) -> deque[float]:
        window = self._windows.get(key)
        cutoff = now - self.window_seconds
        if window is None:
            window = deque()
            self._windows[key] = window
        else:
            while window and window[0] <= cutoff:
                window.popleft()
        return window

    def allow(self, key: Hashable) -> bool:
        """Return True if a request for ``key`` is within the limit."""
        now = time.monotonic()
        with self._lock:
            window = self._requests_in_window(key, now)
            if len(window) >= self.max_requests:
                return False
            window.append(now)
            return True

    def check(self, key: Hashable, retry_after: float | None = None) -> None:
        """Raise ``RateLimitExceededError`` if ``key`` is over the limit."""
        if not self.allow(key):
            raise RateLimitExceededError(
                f"Rate limit exceeded: max {self.max_requests} requests "
                f"per {self.window_seconds:.0f}s",
                retry_after=retry_after if retry_after is not None else self.window_seconds,
            )

    def reset(self, key: Hashable | None = None) -> None:
        """Clear history for a single key or for all keys."""
        with self._lock:
            if key is None:
                self._windows.clear()
            else:
                self._windows.pop(key, None)
