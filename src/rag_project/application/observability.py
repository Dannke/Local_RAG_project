"""Request correlation and timing helpers shared by use cases and adapters.

A per-request id lives in a ``ContextVar`` so every log line emitted while
answering one question can be correlated; the logging filter in
``rag_project.logging_setup`` reads it.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("rag_request_id", default="")


def new_request_id() -> str:
    """Create (and bind) a fresh request id for the current context."""
    request_id = uuid.uuid4().hex[:12]
    request_id_var.set(request_id)
    return request_id


def now_ms() -> int:
    """Monotonic clock in milliseconds, for measuring latencies."""
    return int(time.monotonic() * 1000)
