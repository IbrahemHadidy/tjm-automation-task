"""Provide core infrastructure for environment config, logging, and resilience.

Manage cross-cutting concerns including environment variable loading,
structured JSON telemetry, and execution retry logic with exponential backoff.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import random
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Callable
    from enum import Enum


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
# PROCESS ENVIRONMENT INITIALIZATION
# ------------------------------------------------------------


def set_high_dpi_awareness() -> None:
    """Configure the current process to be DPI-aware on Windows.

    This ensures that all screen coordinates (screenshots, mouse input,
    window geometry) operate on the physical pixel grid instead of
    scaled logical coordinates.

    Fallback strategies are applied to support different Windows APIs.
    """
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4  # noqa: N806

    logger = build_logger()

    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
        )
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception as e:
                logger.warning("Failed to set DPI awareness: %s", e)


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

    def debug(self, message: str, **data: object) -> None:
        """Log a DEBUG level message with structured metadata."""
        self.logger.debug(json.dumps({"level": "DEBUG", "message": message, **data}))

    def info(self, message: str, **data: object) -> None:
        """Log an INFO level message with structured metadata."""
        self.logger.info(json.dumps({"level": "INFO", "message": message, **data}))

    def warning(self, message: str, **data: object) -> None:
        """Log a WARN level message with structured metadata."""
        self.logger.warning(json.dumps({"level": "WARN", "message": message, **data}))

    def error(self, message: str, **data: object) -> None:
        """Log an ERROR level message with structured metadata."""
        self.logger.error(json.dumps({"level": "ERROR", "message": message, **data}))

    def exception(self, message: str, **data: object) -> None:
        """Log an ERROR level message with traceback and metadata.

        Fixes the 'No parameter named "error"' CallIssue in Pyright.
        """
        self.logger.exception(
            json.dumps({"level": "EXCEPTION", "message": message, **data}),
        )


def build_logger(name: str = "automation") -> logging.Logger:
    """Construct a rotating file logger for the automation pipeline.

    Ensures that handlers are only added once to the logger singleton to
    prevent duplicate log entries. It also disables propagation to prevent
    child loggers from doubling up via the parent.

    Args:
        name: The namespace for the logger. Defaults to "automation".

    Returns:
        A configured logging.Logger instance.

    """
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Singleton check: If handlers exist, we've already configured this logger
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Critical to stop double logging from children

        file_handler = RotatingFileHandler(
            LOG_DIR / "automation.log",
            maxBytes=1_000_000,
            backupCount=3,
        )
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# ------------------------------------------------------------
# MONITORING COMPONENTS
# ------------------------------------------------------------


class HeartbeatMonitor:
    """Daemon thread that emits 'Liveness' signals during long-running tasks."""

    def __init__(self, logger: StructuredLogger, interval: int = 30) -> None:
        """Initialize the heartbeat monitor with a logger and pulse frequency.

        Args:
            logger: The custom structured logger used to emit liveness signals.
            interval: Time in seconds between each heartbeat pulse.

        """
        self.logger = logger
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Launch the heartbeat thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the heartbeat thread."""
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.logger.info(
                "SYSTEM_HEARTBEAT | Status: Healthy | Resource Usage: Normal",
            )
            time.sleep(self.interval)


class TaskWatchdog:
    """Monitors Task State transitions to prevent infinite hangs.

    If the state does not change within the timeout period, the watchdog
    triggers a hard exit to prevent zombie processes.
    """

    def __init__(self, timeout: float, check_interval: float = 5.0) -> None:
        """Initialize the watchdog with a stagnation limit and polling frequency.

        Args:
            timeout: Maximum seconds allowed in a single state before termination.
            check_interval: Frequency in seconds at which the watchdog evaluates
                the elapsed time since the last transition.

        """
        self.timeout = timeout
        self.check_interval = check_interval
        self._last_state: Enum | None = None
        self._last_transition_time = time.perf_counter()
        self._stop_event = threading.Event()

    def update_state(self, new_state: Enum) -> None:
        """Signal that the FSM has moved to a new state."""
        self._last_state = new_state
        self._last_transition_time = time.perf_counter()

    def start(self) -> None:
        """Start monitoring for state stagnation."""
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            elapsed = time.perf_counter() - self._last_transition_time
            if elapsed > self.timeout:
                print(
                    f"[CRITICAL] Watchdog triggered! State '{self._last_state}' hung for {elapsed:.1f}s",
                )

                os._exit(1)  # Hard exit to prevent zombie execution
            time.sleep(self.check_interval)


# ------------------------------------------------------------
# RESILIENCE
# ------------------------------------------------------------


def retry[T](
    operation: Callable[[], T],
    attempts: int = RETRY_ATTEMPTS,
    label: str = "Unspecified Operation",
) -> T:
    """Execute a synchronous operation with exponential backoff and structured logging.

    This function implements a resilient execution pattern by catching exceptions,
    logging the failure context using the StructuredLogger, and waiting for an
    increasing amount of time before subsequent attempts.

    Args:
        operation: The callable to be executed.
        attempts: The maximum number of execution tries allowed. Defaults to
            the RETRY_ATTEMPTS constant.
        label: A human-readable identifier for the operation used in telemetry
            to track specific points of failure.

    Returns:
        The result of the successfully executed operation.

    Raises:
        RuntimeError: If the maximum number of attempts is reached or if the
            retry loop exits unexpectedly.
        Exception: The original exception from the final attempt is re-raised
            if all retries fail.

    """
    logger = StructuredLogger(build_logger())

    for attempt in range(attempts):
        try:
            if attempt > 0:
                logger.info(
                    "Attempting recovery",
                    operation=label,
                    current_attempt=attempt + 1,
                    max_attempts=attempts,
                )

            return operation()

        except Exception as e:
            if attempt == attempts - 1:
                logger.error(  # noqa: TRY400
                    "Operation exhausted all retry attempts",
                    operation=label,
                    error=str(e),
                    status="FATAL",
                )
                raise

            # Apply exponential backoff: 2^attempt + jitter
            wait_time = (2**attempt) + random.uniform(0, 0.3)  # noqa: S311

            logger.warning(
                "Operation failed, initiating backoff",
                operation=label,
                attempt=attempt + 1,
                next_retry_in_sec=round(wait_time, 2),
                error_type=type(e).__name__,
                error_message=str(e),
            )

            time.sleep(wait_time)

    msg = f"Retry limit reached unexpectedly for: {label}"
    raise RuntimeError(msg)
