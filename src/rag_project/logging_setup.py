"""Structured logging setup and request (trace) context helpers.

The app logs a JSON record per event when ``RAG_STRUCTURED_LOGGING`` is
enabled, which is useful for collecting metrics / eval data on a VPS. A
per-request id is propagated via ``contextvars`` so every log line emitted
while answering one question can be correlated in a log aggregator.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

RequestIdVar: ContextVar[str] = ContextVar("rag_request_id", default="")

_RESERVED_ATTRS = frozenset(
    {
        "args",
        "msg",
        "exc_info",
        "exc_text",
        "stack_info",
        "filename",
        "lineno",
        "module",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "levelno",
        "pathname",
        "name",
        "rag_request_id",
        "epoch_ms",
    }
)


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
            if key.startswith("_") or key in _RESERVED_ATTRS:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(console_level: int = logging.INFO, file_level: int = logging.INFO) -> None:
    """Configure root logging for either structured JSON or readable text.

    Controlled by ``RAG_STRUCTURED_LOGGING`` (default off: human-readable, on:
    one JSON line per event). A ``RotatingFileHandler`` is added alongside the
    console handler so logs are persisted and rotated on the VPS. The file is
    written to ``RAG_LOG_DIR`` (default ``logs/app.log``) and rotated when it
    reaches ``RAG_LOG_MAX_BYTES`` keeping ``RAG_LOG_BACKUP_COUNT`` backups.

    Re-invoking after a first call is a no-op.
    """
    if getattr(setup_logging, "_done", False):
        return

    formatter = _build_formatter()

    console = logging.StreamHandler()
    console.addFilter(RequestIdFilter())
    console.setFormatter(formatter)
    console.setLevel(console_level)

    handlers: list[logging.Handler] = [console]

    log_dir = Path(os.environ.get("RAG_LOG_DIR", "logs"))
    max_bytes = int(os.environ.get("RAG_LOG_MAX_BYTES", "10_000_000"))
    backup_count = int(os.environ.get("RAG_LOG_BACKUP_COUNT", "5"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.addFilter(RequestIdFilter())
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        handlers.append(file_handler)
    except OSError:
        # Fall back to console-only if the log dir is not writable.
        pass

    root = logging.getLogger()
    if not root.handlers:
        for handler in handlers:
            root.addHandler(handler)
    root.setLevel(min(console_level, file_level))

    setup_logging._done = True  # type: ignore[attr-defined]


def _build_formatter() -> logging.Formatter:
    if os.environ.get("RAG_STRUCTURED_LOGGING", "").lower() in {"1", "true", "yes", "on"}:
        return JsonFormatter()
    return logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [request=%(rag_request_id)s] %(message)s"
    )


def new_request_id() -> str:
    """Create (and bind) a fresh request id for the current async context."""
    request_id = uuid.uuid4().hex[:12]
    RequestIdVar.set(request_id)
    return request_id


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
