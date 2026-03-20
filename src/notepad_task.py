"""FSM orchestration controller for production-grade automation pipelines.

This module manages the full lifecycle of the automation task via Finite State
Machine (FSM) transitions while delegating UI operations to specialized drivers.
It maintains strict workspace isolation and provides detailed telemetry.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import TYPE_CHECKING, TypedDict

import cv2
import pyautogui
import pygetwindow as gw
import requests

from config import (
    API_TIMEOUT_SEC,
    API_URL,
    GLOBAL_TIMEOUT_SEC,
    ICON_PATH,
    MAX_POSTS,
    PROJECT_DIR,
    TESS_PATH,
)
from core import HeartbeatMonitor, StructuredLogger, TaskWatchdog, build_logger, retry
from drivers import NotepadDriver
from monitoring import RunMetrics, RunMonitor
from screenshot_service import ScreenshotService
from utils import (
    check_for_interference,
    get_pid_from_hwnd,
    kill_bot_process_only,
    preserve_clipboard,
    prevent_system_sleep,
    validate_environment,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from strategies import LaunchStrategy


class TaskState(Enum):
    """Enumeration of all possible states in the automation lifecycle.

    Attributes:
        INIT: System setup, health checks, and safety snapshots.
        FETCH_POSTS: Data acquisition from the external REST API.
        PROCESS_POST: Main loop control and bounds checking.
        LAUNCH: Application perception and window handle acquisition.
        WRITE: Input content into the editor.
        SAVE: Persist content to disk via Save dialog.
        CLOSE: Close the application window.
        NEXT_POST: Iterator increment and pointer management.
        RESTORE: Returning the desktop environment to the user.
        DONE: Successful completion of all requested tasks.
        FAILED: Error handling path for recovery or exit.

    """

    INIT = auto()
    FETCH_POSTS = auto()
    PROCESS_POST = auto()
    LAUNCH = auto()
    WRITE = auto()
    SAVE = auto()
    CLOSE = auto()
    NEXT_POST = auto()
    RESTORE = auto()
    DONE = auto()
    FAILED = auto()


class Post(TypedDict):
    """Schema representing posts returned from the external API.

    Attributes:
        userId: The unique identifier of the user who created the post.
        id: The unique identifier of the post itself.
        title: The headline or subject of the post.
        body: The main content/narrative of the post.

    """

    userId: int
    id: int
    title: str
    body: str


class NotepadTask:
    """Orchestrates the Notepad automation task using a Finite State Machine.

    This controller manages the synchronization between visual perception strategies
    and low-level UI drivers. It ensures that bot-created windows do not interfere
    with existing user windows by tracking window handles (HWNDs).

    Attributes:
        strategy (LaunchStrategy): The perception engine used to locate the app.
        logger (StructuredLogger): JSON-structured telemetry emitter.
        monitor (RunMonitor): Orchestrator for screenshots and run manifests.
        state (TaskState): The current position within the FSM.
        metrics (dict): Performance and success counters for the run.

    """

    MAX_LAUNCH_ATTEMPTS = 3

    def __init__(self, strategy: LaunchStrategy) -> None:
        """Initialize the task runner with grounding and background safety monitors."""
        self.strategy = strategy
        self.logger = StructuredLogger(build_logger())
        self.monitor = RunMonitor(PROJECT_DIR)
        RunMonitor.current_run_dir = self.monitor.run_dir
        self.screen_service = ScreenshotService(settle_delay=1.0)

        # FSM State
        self.state = TaskState.INIT
        self.state_start_time = time.perf_counter()
        self.total_run_start_time = time.perf_counter()

        # Monitoring Pieces
        self.heartbeat = HeartbeatMonitor(self.logger, interval=30)
        self.watchdog = TaskWatchdog(timeout=GLOBAL_TIMEOUT_SEC)

        # Data & Resource Tracking
        self.posts: list[Post] = []
        self.current_post_index = 0
        self.user_notepad_handles: set[int] = set()
        self.driver: NotepadDriver | None = None
        self.bot_pid: int | None = None
        self._last_mouse_pos = pyautogui.position()

        self.metrics: RunMetrics = {
            "processed_count": 0,
            "launch_failures": 0,
            "launch_retries": 0,
            "write_failures": 0,
            "save_failures": 0,
            "close_failures": 0,
            "success_rate": 0.0,
            "step_timings_sec": {},
        }

        self._state_handlers: dict[TaskState, Callable[[], None]] = {
            TaskState.FETCH_POSTS: self._state_fetch_posts,
            TaskState.PROCESS_POST: self._state_process_post,
            TaskState.LAUNCH: self._state_launch,
            TaskState.WRITE: self._state_write,
            TaskState.SAVE: self._state_save,
            TaskState.CLOSE: self._state_close,
            TaskState.NEXT_POST: self._state_next_post,
            TaskState.DONE: self._state_done,
            TaskState.FAILED: self._state_failed,
        }

    def _transition(self, new_state: TaskState) -> None:
        """Execute a state change and notify the background watchdog.

        Args:
            new_state: The TaskState to transition into.

        """
        duration = time.perf_counter() - self.state_start_time
        current_state_name = self.state.name

        # Telemetry updates
        self.metrics["step_timings_sec"][current_state_name] = (
            self.metrics["step_timings_sec"].get(current_state_name, 0.0) + duration
        )

        self.logger.info(
            "state_transition",
            from_state=self.state.name,
            to_state=new_state.name,
            duration_sec=round(duration, 2),
        )

        # UI Step Logging
        self.monitor.log_step(
            "transition",
            new_state.name,
            {"duration": round(duration, 2)},
        )

        # Write the FSM heartbeat to the dedicated pulse file
        self.monitor.log_heartbeat(f"Status: Healthy | FSM State: {new_state.name}")

        # Update State and Watchdog
        self.state = new_state
        self.state_start_time = time.perf_counter()
        self.watchdog.update_state(new_state)  # Reset watchdog timer for the new state
        self._last_mouse_pos = pyautogui.position()

    def _save_strategy_artifact(self) -> None:
        """Extract and save a visual debug frame from the perception engine.

        Provides observability into why the VLM or CV engine chose a
        specific coordinate.
        """
        debug_frame = self.strategy.get_debug_frame()

        if debug_frame is not None:
            artifact_dir = self.monitor.run_dir / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Save the BGR frame from the strategy to the run directory
            save_path = (
                artifact_dir / f"perception_map_post_{self.current_post_index}.png"
            )
            cv2.imwrite(str(save_path), debug_frame)
            self.logger.debug("artifact_saved", path=str(save_path))

    def _archive_existing_outputs(self) -> None:
        """Move existing post files to an archive directory to prevent overwrites."""
        existing_files = list(PROJECT_DIR.glob("post_*.txt"))
        if not existing_files:
            return

        archive_path = (
            PROJECT_DIR / "archives" / f"archive_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            archive_path.mkdir(parents=True, exist_ok=True)
            for f in existing_files:
                f.rename(archive_path / f.name)
        except Exception as e:
            self.logger.error("archive_failed", error=str(e))  # noqa: TRY400

    def _snapshot_user_windows(self) -> None:
        """Capture active Notepad window handles to isolate user data from bot data."""
        self.user_notepad_handles = {w._hWnd for w in gw.getWindowsWithTitle("Notepad")}  # noqa: SLF001

    def _cleanup_bot_windows(self) -> None:
        """Close any Notepad windows created by the bot during the current run."""
        self.logger.info("cleanup_sweeping_bot_windows")
        bot_windows = [
            w
            for w in gw.getWindowsWithTitle("Notepad")
            if w._hWnd not in self.user_notepad_handles  # noqa: SLF001
        ]
        for win in bot_windows.copy():
            try:
                temp_driver = NotepadDriver(win._hWnd, self.logger)  # noqa: SLF001
                temp_driver.close()
            except Exception as e:
                self.logger.debug(
                    "cleanup_skipped_window",
                    hwnd=win._hWnd,  # noqa: SLF001
                    reason=str(e),
                )
                continue

    def _fetch_posts(self) -> list[Post]:
        """Acquire post data from the external API.

        This method performs the raw network request. The retry logic is
        handled by the caller (the FSM run loop) to ensure centralized
        observability.

        Returns:
            A list of Post dictionaries if successful.

        Raises:
            requests.RequestException: If the network call fails after
                the internal timeout.

        """
        resp = requests.get(str(API_URL), timeout=API_TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.json()

    def _save_post(self, post: Post) -> bool:
        """Execute the UI sequence to write and save post content via the driver.

        Args:
            post: The post data to be written to Notepad.

        Returns:
            True if the file was saved successfully, False otherwise.

        """
        if not self.driver:
            return False

        try:
            content = f"Title: {post['title']}\n\n{post['body']}"
            self.driver.write_content(content)

            file_path = str(PROJECT_DIR / f"post_{post['id']}.txt")
            success = self.driver.save_as(file_path)

            if success:
                self.driver.close()
                self.driver = None
        except Exception as e:
            self.logger.error("save_post_failed", error=str(e))  # noqa: TRY400
            return False
        return success

    def _state_fetch_posts(self) -> None:
        """Fetch posts from the API with retry and validation."""
        try:
            self.posts = retry(self._fetch_posts, label="API_POST_FETCH")
        except Exception as e:
            self.logger.error("api_fetch_failed", error=str(e))  # noqa: TRY400
            self.posts = []

        if not self.posts:
            self._transition(TaskState.FAILED)
            return

        self._transition(TaskState.PROCESS_POST)

    def _state_process_post(self) -> None:
        """Check iteration bounds and determine next action."""
        if self.current_post_index >= min(len(self.posts), MAX_POSTS):
            self._transition(TaskState.DONE)
        else:
            self._transition(TaskState.LAUNCH)

    def _state_launch(self) -> None:
        """Launch Notepad using the configured perception strategy."""
        self.logger.info(
            "Preparing desktop for perception using ScreenshotService",
        )

        # 1. Use the service to clear the workspace and click (1, 1)
        self.screen_service.capture_desktop()

        # 2. Execute the launch strategy
        hwnd = retry(self.strategy.launch, label="NOTEPAD_LAUNCH")
        self._save_strategy_artifact()

        if hwnd:
            self.driver = NotepadDriver(hwnd, self.logger)
            self.bot_pid = get_pid_from_hwnd(hwnd)

            # 3. Perform Surgical Masking for Visual Validation
            self.logger.info("Executing surgical mask on target application")

            mask_img, _ = self.screen_service.capture_app_window("Notepad")

            # Save the masked artifact
            artifact_path = (
                self.monitor.run_dir
                / "artifacts"
                / f"surgical_mask_{self.current_post_index}.png"
            )

            mask_img.save(str(artifact_path))

            self._transition(TaskState.WRITE)
        else:
            # If launch fails, restore desktop before failing
            self.metrics["launch_failures"] += 1
            self._transition(TaskState.FAILED)

    def _state_write(self) -> None:
        """Type the post content into the editor (no file dialog)."""
        current_post = self.posts[self.current_post_index]

        if not self.driver:
            self.logger.error("write_no_driver")
            self.metrics["write_failures"] += 1
            self._transition(TaskState.FAILED)
            return

        try:
            content = f"Title: {current_post['title']}\n\n{current_post['body']}"
            # If driver.write_content can be flaky, wrap in retry() as needed.
            self.driver.write_content(content)
            self.logger.info("write_completed", index=current_post.get("id"))
            self._transition(TaskState.SAVE)
        except Exception as e:
            self.logger.error("write_failed", error=str(e))  # noqa: TRY400
            self.metrics["write_failures"] += 1
            # Fail fast or optionally retry — choose what suits your reliability needs
            self._transition(TaskState.FAILED)

    def _state_save(self) -> None:
        """Bring up save dialog / Save As and persist file to disk."""
        current_post = self.posts[self.current_post_index]
        driver = self.driver
        if driver is None:
            self.logger.error("save_no_driver")
            self.metrics["save_failures"] += 1
            self._transition(TaskState.FAILED)
            return

        file_path = str(PROJECT_DIR / f"post_{current_post['id']}.txt")

        try:
            # Save operation is the one most likely to need retry due to dialogs/focus issues
            success = retry(lambda: driver.save_as(file_path), label="FILE_SAVE")
            if success:
                self.logger.info("save_completed", path=file_path)
                self._transition(TaskState.CLOSE)
            else:
                self.logger.error("save_reported_false", path=file_path)
                self.metrics["save_failures"] += 1
                self._transition(TaskState.FAILED)
        except Exception as e:
            self.logger.error("save_exception", error=str(e))  # noqa: TRY400
            self.metrics["save_failures"] += 1
            self._transition(TaskState.FAILED)

    def _state_close(self) -> None:
        """Close the editor window and do lightweight cleanup."""
        current_post = self.posts[self.current_post_index]

        if not self.driver:
            self.logger.warning("close_no_driver")
            # still advance to next post to avoid getting stuck
            self._transition(TaskState.NEXT_POST)
            return

        try:
            # Attempt a graceful close. If it fails, log and attempt forced cleanup later.
            self.driver.close()
            self.driver = None
            self.logger.info("close_completed", index=current_post.get("id"))
            self.metrics["processed_count"] += 1
            self._transition(TaskState.NEXT_POST)
        except Exception as e:
            self.logger.error("close_failed", error=str(e))  # noqa: TRY400
            self.metrics["close_failures"] += 1

            # Try more aggressive cleanup to avoid leaving zombie windows
            try:
                if self.bot_pid:
                    kill_bot_process_only(self.bot_pid)
            except Exception:
                self.logger.debug("kill_bot_failed", error=str(e))

            # continue to NEXT_POST to keep the run progressing
            self.driver = None
            self._transition(TaskState.NEXT_POST)

    def _state_next_post(self) -> None:
        """Increment post pointer."""
        self.current_post_index += 1
        self._transition(TaskState.PROCESS_POST)

    def _state_done(self) -> None:
        """Handle successful completion of the FSM."""
        self.logger.info(
            "task_completed",
            processed=self.metrics["processed_count"],
        )

    def _state_failed(self) -> None:
        """Handle task failure state."""
        self.logger.error(
            "task_failed",
            processed=self.metrics["processed_count"],
        )

    def _handle_terminal_state(self) -> bool:
        """Check and process terminal FSM states immediately."""
        if self.state == TaskState.DONE:
            self._state_done()
            return True
        if self.state == TaskState.FAILED:
            self._state_failed()
            return True
        return False

    def run(self) -> None:
        """Execute the full automation lifecycle using a Finite State Machine (FSM).

        This method orchestrates the entire task lifecycle, ensuring safe execution,
        deterministic cleanup, and detailed telemetry. The FSM loop delegates operational
        work to dedicated state handlers (`_state_*` methods) while this method manages
        global lifecycle concerns such as monitoring, safety controls, and resource restoration.

        Execution Steps:
            1. Capture the current desktop layout for workspace restoration.
            2. Validate environment assets and required executables.
            3. Prevent system sleep to avoid visual grounding failure.
            4. Archive old outputs and snapshot user Notepad windows.
            5. Start background monitoring systems (heartbeat + watchdog).
            6. Execute the FSM loop with interference detection.
            7. Handle terminal states (DONE or FAILED) immediately.
            8. Perform deterministic cleanup and telemetry finalization.
        """
        try:
            # 1. Desktop State Persistence
            self.screen_service.snapshot_workspace()

            # 2. Environment & Asset Integrity Check
            validate_environment(
                required_assets=[ICON_PATH],
                tesseract_path=TESS_PATH,
            )
            prevent_system_sleep(
                True,  # Prevent OS sleep/lock during automation # noqa: FBT003
            )

            # 3. Process & Window Isolation
            self._archive_existing_outputs()
            self._snapshot_user_windows()
            self._cleanup_bot_windows()

            # 4. Telemetry & Safety Watchdogs
            self.heartbeat.start()
            self.watchdog.start()

            # 5. Secure Execution Loop
            with preserve_clipboard():
                self._transition(TaskState.FETCH_POSTS)

                # Core FSM execution loop
                while True:
                    # Immediate exit if terminal state is reached
                    if self._handle_terminal_state():
                        break

                    # Detect user interference to avoid fighting for desktop control
                    check_for_interference(
                        self._last_mouse_pos,
                        screen_service=self.screen_service,
                        logger=self.logger,
                    )

                    # Dispatch handler for the current state
                    handler = self._state_handlers.get(self.state)
                    if handler is None:
                        self.logger.error("no_handler_for_state", state=self.state.name)
                        self._transition(TaskState.FAILED)
                        continue

                    handler()  # Execute the state handler

        except Exception as e:
            self.monitor.log_error(e)
            self.logger.exception("task_execution_aborted", error=str(e))
            self._transition(TaskState.FAILED)

        finally:
            # Restore workspace and system state
            self.screen_service.restore_workspace()
            prevent_system_sleep(False)  # noqa: FBT003
            self.heartbeat.stop()
            self.watchdog.stop()
            self._cleanup_bot_windows()
            kill_bot_process_only(self.bot_pid)

            runtime = time.perf_counter() - self.total_run_start_time
            total_attempts = (
                self.metrics["processed_count"]
                + self.metrics["launch_failures"]
                + self.metrics["save_failures"]
            )

            # Calculate final performance KPIs before persisting the manifest.
            # This ensures the success_rate reflects the outcome of all FSM transitions.
            if total_attempts > 0:
                self.metrics["success_rate"] = round(
                    self.metrics["processed_count"] / total_attempts,
                    2,
                )

            # Persist run metadata and generate visual replay artifacts
            self.monitor.compile_video_summary()
            self.monitor.finalize(self.metrics)

            self.logger.info(
                "fsm_exit",
                state=self.state.name,
                runtime_sec=round(runtime, 2),
            )
