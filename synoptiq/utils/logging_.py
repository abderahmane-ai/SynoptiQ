"""Structured logging utilities for SynoptiQ.

Uses stdlib logging with a JSON formatter for structured output.
Optionally initializes Weights & Biases if WANDB_API_KEY is set.

Intentionally minimal: training scripts call ``get_logger(__name__)`` and log
JSON lines to stdout. W&B receives metrics via ``wandb.log()`` calls in the
training loops themselves; this module only handles process-level setup.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import os
import sys
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Formats log records as JSON lines for structured log consumption."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as a single JSON line.

        Emits the timestamp, level, logger name, and message, plus the
        traceback for records carrying exception info and any extra fields
        passed via ``logger.log(..., extra={...})``.
        """
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any extra kwargs passed to the logger
        for key, val in record.__dict__.items():
            if key not in {
                "args",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "message",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "taskName",
                "thread",
                "threadName",
            }:
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger configured with JSON output to stdout.

    Idempotent: calling multiple times with the same name returns the
    same logger without adding duplicate handlers.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured Logger instance.

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("corpus loaded", extra={"n_tokens": 138000})
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def setup_wandb(
    project: str,
    config: dict[str, Any] | None = None,
    *,
    run_name: str | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Initialize a Weights & Biases run if WANDB_API_KEY is set.

    No-ops silently when WANDB_API_KEY is absent (e.g., in CI or local dev).
    Returns the wandb run object, or None if W&B is not available/configured.

    Args:
        project: W&B project name.
        config: Dictionary of hyperparameters to log.
        run_name: Optional run display name.
        tags: Optional list of tags.

    Returns:
        The wandb.Run object, or None if W&B is not initialized.

    Example:
        >>> run = setup_wandb("synoptiq", {"lr": 5e-5, "phase": "DAPT"})
    """
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        log = get_logger(__name__)
        log.info("WANDB_API_KEY not set — W&B logging disabled")
        return None

    try:
        import wandb  # type: ignore[import-untyped]

        run = wandb.init(
            project=project,
            config=config or {},
            name=run_name,
            tags=tags or [],
            reinit=True,
        )
        return run
    except ImportError:
        log = get_logger(__name__)
        log.warning("wandb not installed — W&B logging disabled")
        return None
