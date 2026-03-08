"""Provide per-run observability and telemetry management.

Organize screenshots, execution metadata, and logs into a deterministic,
timestamped directory structure to allow for post-run audits and debugging.

Directory Structure:
    logs/
    └── run_YYYYMMDD_HHMMSS_uuid/
        ├── metadata.json       # Full run telemetry and metrics
        ├── steps/              # Screenshots for every FSM transition
        ├── errors/             # Screenshots and tracebacks for failures
        └── artifacts/          # Manually captured debug images
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import pyautogui

if TYPE_CHECKING:
    from pathlib import Path


class RunMonitor:
    """Manage automation telemetry and visual evidence collection.

    Create a unique sandbox directory for each execution run and provide
    utilities to capture the desktop state without interrupting main logic.

    Attributes:
        current_run_dir (Path | None): Access the active log directory globally.

    """

    # Class-level reference allows other modules to find the current log path
    current_run_dir = None

    def __init__(self, base_dir: Path) -> None:
        """Initialize the monitor and prepare the physical directory tree.

        Args:
            base_dir: Root path where the 'logs' folder will be managed.

        """
        self.logs_root = base_dir / "logs"
        self.run_id = self._generate_run_id()

        self.run_dir = self.logs_root / self.run_id
        self.steps_dir = self.run_dir / "steps"
        self.error_dir = self.run_dir / "errors"
        self.artifacts_dir = self.run_dir / "artifacts"

        RunMonitor.current_run_dir = self.run_dir

        for folder in (
            self.run_dir,
            self.steps_dir,
            self.error_dir,
            self.artifacts_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        self.execution_metadata: dict[str, Any] = {
            "run_id": self.run_id,
            "start_time_unix": time.time(),
            "steps": [],
        }

    def _generate_run_id(self) -> str:
        """Generate a unique, sortable ID for the execution folder.

        Returns:
            A string formatted as 'run_YYYYMMDD_HHMMSS_uniquehash'.

        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        return f"run_{timestamp}_{unique_suffix}"

    def _safe_screenshot(self, save_path: str) -> None:
        """Capture the screen while ensuring the main pipeline never crashes.

        Args:
            save_path: Absolute system path for the PNG file.

        """
        with suppress(Exception):
            pyautogui.screenshot(save_path)

    def log_step(
        self,
        step_name: str,
        fsm_state: str,
        context_data: dict[str, Any] | None = None,
    ) -> None:
        """Record a state transition and save a grounding screenshot.

        Args:
            step_name: Descriptive name of the action.
            fsm_state: Current state name from the TaskState enum.
            context_data: Optional metadata like execution time or API results.

        """
        step_number = len(self.execution_metadata["steps"]) + 1
        img_filename = f"step_{step_number:03d}_{step_name}.png"
        img_path = self.steps_dir / img_filename

        self._safe_screenshot(str(img_path))

        step_record: dict[str, Any] = {
            "index": step_number,
            "action": step_name,
            "fsm_state": fsm_state,
            "timestamp": time.time(),
            "screenshot_path": str(img_path),
        }

        if context_data:
            step_record["context"] = context_data

        self.execution_metadata["steps"].append(step_record)

    def capture_screenshot(self, label: str) -> None:
        """Capture a manual artifact screenshot for debugging.

        Args:
            label: Filename prefix for the screenshot.

        """
        save_path = self.artifacts_dir / f"{label}.png"
        self._safe_screenshot(str(save_path))

    def log_error(self, error: Exception) -> None:
        """Record a fatal exception and capture the screen at moment of failure.

        Args:
            error: Exception object caught by the main loop.

        """
        unix_ts = int(time.time())
        crash_img_path = self.error_dir / f"crash_{unix_ts}.png"

        self._safe_screenshot(str(crash_img_path))

        self.execution_metadata["error_details"] = {
            "exception_type": type(error).__name__,
            "message": str(error),
            "timestamp": time.time(),
            "crash_screenshot": str(crash_img_path),
        }

    def finalize(self, run_metrics: dict[str, Any]) -> None:
        """Write the final telemetry manifest to a JSON file.

        Args:
            run_metrics: Summary data like success rate or total items processed.

        """
        self.execution_metadata["end_time_unix"] = time.time()
        self.execution_metadata["summary_metrics"] = run_metrics

        manifest_path = self.run_dir / "metadata.json"
        manifest_path.write_text(
            json.dumps(self.execution_metadata, indent=2),
        )
