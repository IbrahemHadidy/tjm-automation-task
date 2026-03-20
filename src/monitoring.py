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
from pathlib import Path
from typing import Any, TypedDict

import cv2
import pyautogui


class RunMetrics(TypedDict):
    """Structured telemetry metrics captured during a task execution.

    This schema defines the summary statistics recorded at the end of an
    automation run and persisted in the run metadata manifest.

    The metrics provide visibility into FSM performance, failure modes,
    and execution timing for each automation phase.

    Attributes:
        processed_count:
            Total number of posts successfully written and saved.

        launch_failures:
            Number of failed attempts to launch the target application.

        launch_retries:
            Number of retry attempts triggered during launch operations.

        write_failures:
            Number of failures while writing post content to the editor.

        save_failures:
            Number of failures encountered while saving files to disk.

        close_failures:
            Number of failures encountered while closing the application
            window after a save operation.

        success_rate:
            Ratio of successfully processed posts to total attempted posts.

        step_timings_sec:
            Aggregated execution time per FSM state in seconds.
            Keys correspond to the state names from the TaskState enum.

    """

    processed_count: int
    launch_failures: int
    launch_retries: int
    write_failures: int
    save_failures: int
    close_failures: int
    success_rate: float
    step_timings_sec: dict[str, float]


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

    def log_heartbeat(self, message: str) -> None:
        """Emit a periodic 'alive' signal with current state metrics.

        Useful for identifying infinite loops or unhandled UI deadlocks.

        Args:
            message: A descriptive status message.

        """
        # We write to a separate heartbeat file to avoid cluttering metadata
        heartbeat_path = self.run_dir / "heartbeat.log"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with Path.open(heartbeat_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def compile_video_summary(self, fps: int = 2) -> None:
        """Stitch all captured step screenshots into an MP4 video payload.

        Args:
            fps: Frames per second for the resulting video. Defaults to 2.

        """
        images = sorted(self.steps_dir.glob("*.png"))
        if not images:
            return

        video_path = str(self.run_dir / "execution_replay.mp4")

        # Read first image to get dimensions
        frame = cv2.imread(str(images[0]))
        if frame is None:
            msg = f"Failed to load image: {images[0]}"
            raise RuntimeError(msg)

        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        print("[TELEMETRY] Compiling execution video replay...")
        for image_path in images:
            img = cv2.imread(str(image_path))
            if img is not None:
                # Resize if necessary to match the first frame
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                video.write(img)

        video.release()

    def finalize(self, run_metrics: RunMetrics) -> None:
        """Write the final telemetry manifest to a JSON file.

        Args:
            run_metrics: Summary data like success rate or total items processed.

        """
        self.execution_metadata["end_time_unix"] = time.time()
        self.execution_metadata["summary_metrics"] = run_metrics

        video_path = self.run_dir / "execution_replay.mp4"
        if video_path.exists():
            self.execution_metadata["execution_video"] = str(video_path)

        manifest_path = self.run_dir / "metadata.json"
        manifest_path.write_text(
            json.dumps(self.execution_metadata, indent=2),
        )
