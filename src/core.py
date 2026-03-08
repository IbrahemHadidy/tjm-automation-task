"""Provide core infrastructure for environment config, logging, and resilience.

Manage cross-cutting concerns including environment variable loading,
structured JSON telemetry, and execution retry logic with exponential backoff.
"""

from __future__ import annotations

import json
import logging
import random
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Callable

# ------------------------------------------------------------
# ENV + CONFIG
# ------------------------------------------------------------

load_dotenv()

MAX_POSTS = 10
RETRY_ATTEMPTS = 3
API_TIMEOUT_SEC = 10

PROJECT_DIR = Path.home() / "Desktop" / "tjm-project"
LOG_DIR = PROJECT_DIR / "logs"


# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------


class StructuredLogger:
    """Adapt standard logging to emit JSON-serializable structured data.

    Enforce a consistent telemetry format for downstream log parsers and
    observability dashboards.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize with a standard Python logger instance.

        Args:
            logger: The underlying logging.Logger to use.

        """
        self.logger = logger

    def info(self, message: str, **data: object) -> None:
        """Log an INFO level message with structured metadata."""
        self.logger.info(json.dumps({"level": "INFO", "message": message, **data}))

    def warning(self, message: str, **data: object) -> None:
        """Log a WARN level message with structured metadata."""
        self.logger.warning(json.dumps({"level": "WARN", "message": message, **data}))

    def error(self, message: str, **data: object) -> None:
        """Log an ERROR level message with structured metadata."""
        self.logger.error(json.dumps({"level": "ERROR", "message": message, **data}))

    def exception(self, message: str) -> None:
        """Log an ERROR level message with the current exception traceback."""
        self.logger.exception(message)


def build_logger() -> logging.Logger:
    """Construct a rotating file logger for the automation pipeline.

    Returns:
        A configured logging.Logger instance pointing to LOG_DIR.

    """
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("automation")
    logger.setLevel(logging.INFO)

    # 1. Create the File Handler (Existing)
    file_handler = RotatingFileHandler(
        LOG_DIR / "automation.log",
        maxBytes=1_000_000,
        backupCount=3,
    )

    # 2. Create the Console Handler
    console_handler = logging.StreamHandler()

    # 3. Create the Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Apply formatter to both handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ------------------------------------------------------------
# RESILIENCE
# ------------------------------------------------------------


def retry[T](operation: Callable[[], T], attempts: int = RETRY_ATTEMPTS) -> T:
    """Execute a synchronous operation with exponential backoff retries.

    Args:
        operation: The function to execute.
        attempts: Maximum number of retries before raising.

    Returns:
        The result of the operation if successful.

    Raises:
        RuntimeError: If all retry attempts are exhausted without success.

    """
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt == attempts - 1:
                # Re-raise the original exception on the final attempt
                raise

            # Apply exponential backoff: 2^attempt + jitter
            wait_time = (2**attempt) + random.uniform(0, 0.3)  # noqa: S311
            time.sleep(wait_time)

    # Fallback in case of unexpected loop exit
    msg = "Retry limit reached unexpectedly"
    raise RuntimeError(msg)
