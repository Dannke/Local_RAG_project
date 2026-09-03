"""Structured logging setup and request (trace) context helpers.

The app logs a JSON record per event when ``RAG_STRUCTURED_LOGGING`` is
enabled, which is useful for collecting metrics / eval data on a VPS. A
per-request id is propagated via ``contextvars`` so every log line emitted
while answering one question can be correlated in a log aggregator.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar

RequestIdVar: ContextVar[str] = ContextVar("rag_request_id", default="")


class RequestIdFilter(logging.Filter):
    """Add the current request id and timestamps to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.rag_request_id = RequestIdVar.get() or "-"
        record.epoch_ms = int(time.time() * 1000)
        return True


class JsonFormatter(logging.Formatter):
    """Format each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": record.epoch_ms,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": record.rag_request_id,
        }
        for key, value in record.__dict__.items():
            if key in {"ts", "level", "logger", "message", "request_id"}:
                continue
            if key.startswith("_") or key in {"args", "msg", "exc_info", "exc_text", "stack_info", "filename", "lineno", "module", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "taskName", "levelno", "pathname", "name", "rag_request_id", "epoch_ms"}:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """Configure root logging for either structured JSON or readable text.

    Controlled by ``RAG_STRUCTURED_LOGGING`` (default off: human-readable).
    Re-invoking after a first call is a no-op.
    """
    if getattr(setup_logging, "_done", False):
        return

    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())

    if os.environ.get("RAG_STRUCTURED_LOGGING", "").lower() in {"1", "true", "yes", "on"}:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s [request=%(rag_request_id)s] %(message)s"
            )
        )

    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)

    setup_logging._done = True  # type: ignore[attr-defined]


def new_request_id() -> str:
    """Create (and bind) a fresh request id for the current async context."""
    request_id = uuid.uuid4().hex[:12]
    RequestIdVar.set(request_id)
    return request_id


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
