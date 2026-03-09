"""FSM orchestration controller for production-grade automation pipelines.

This module manages the full lifecycle of the automation task:
1. Environment preparation (archiving old data, snapshotting user windows).
2. Data acquisition from external APIs.
3. UI automation via Finite State Machine (FSM) transitions.
4. Clean teardown and workspace restoration.
"""

from __future__ import annotations

import contextlib
import os
import time
from enum import Enum, auto
from typing import TYPE_CHECKING, TypedDict, cast

import cv2
import pyautogui
import pygetwindow as gw
import pyperclip
import requests

from core import (
    API_TIMEOUT_SEC,
    MAX_POSTS,
    PROJECT_DIR,
    StructuredLogger,
    build_logger,
    retry,
)
from monitoring import RunMonitor

if TYPE_CHECKING:
    from strategies import LaunchStrategy


class TaskState(Enum):
    """Enumeration of all possible states in the automation lifecycle."""

    INIT = auto()  # System setup and safety snapshots
    FETCH_POSTS = auto()  # Acquiring data from API
    PROCESS_POST = auto()  # Loop control logic
    LAUNCH = auto()  # Application perception and startup
    SAVE_AND_CLOSE = auto()  # UI interaction and file verification
    NEXT_POST = auto()  # Iterator increment
    RESTORE = auto()  # Returning desktop to user
    DONE = auto()  # Successful exit
    FAILED = auto()  # Error handling path


class Post(TypedDict):
    """Schema representing posts returned from the external API."""

    userId: int
    id: int
    title: str
    body: str


class NotepadTask:
    """Orchestrates the Notepad automation task using an FSM.

    Maintains strict isolation between bot activity and user activity by
    tracking window handles and snapshotting the desktop state.
    """

    MAX_LAUNCH_ATTEMPTS = 3
    GLOBAL_TIMEOUT_SEC = 900  # 15-minute watchdog timer

    def __init__(self, strategy: LaunchStrategy) -> None:
        """Initialize the task runner with grounding and safety features.

        Args:
            strategy: The perception strategy (OpenCV/VLM) used to locate the app.

        """
        self.strategy = strategy
        self.logger = StructuredLogger(build_logger())
        self.monitor = RunMonitor(PROJECT_DIR)

        # Safety: Small pause between pyautogui actions prevents 'input spam'
        pyautogui.PAUSE = 0.1
        pyautogui.FAILSAFE = True

        # Link the global run directory for artifact access
        RunMonitor.current_run_dir = self.monitor.run_dir

        # FSM State Tracking
        self.state = TaskState.INIT
        self.state_start_time = time.perf_counter()
        self.total_run_start_time = time.perf_counter()

        self.posts: list[Post] = []
        self.current_post_index = 0
        self.original_workspace_snapshot: list = []

        # Isolation: Use Window Handles (_hWnd) as unique identifiers
        self.user_notepad_handles: set[int] = set()

        api_url = os.getenv("API_URL")
        if not api_url:
            msg = "API_URL missing from environment variables"
            raise ValueError(msg)
        self.api_url = api_url

        self.metrics = {
            "processed_count": 0,
            "launch_failures": 0,
            "launch_retries": 0,
            "success_rate": 0.0,
        }

    # =========================================================
    # SAFETY & COEXISTENCE UTILITIES
    # =========================================================

    def _archive_existing_outputs(self) -> None:
        """Move existing txt files to a uniquely timestamped archive folder.

        Why: Prevents the bot from overwriting previous results or getting
        confused by existing 'post_*.txt' files in the workspace.
        """
        existing_files = list(PROJECT_DIR.glob("post_*.txt"))
        if not existing_files:
            return

        timestamp = time.strftime("%Y-%m-%d_%H%M%S")
        archive_path = PROJECT_DIR / "archives" / f"archive_{timestamp}"

        try:
            archive_path.mkdir(parents=True, exist_ok=True)
            self.monitor.log_step(
                "environment_cleanup",
                self.state.name,
                {"archived_to": str(archive_path), "file_count": len(existing_files)},
            )

            for file_path in existing_files:
                file_path.rename(archive_path / file_path.name)

            self.logger.info("archiving_complete", destination=str(archive_path))
        except Exception as e:
            self.logger.error("archive_failed", error=str(e))  # noqa: TRY400
            self.monitor.capture_screenshot("archive_fatal_error")

    def _snapshot_user_windows(self) -> None:
        """Record IDs of Notepad windows open BEFORE the bot starts.

        This ensures the bot never types into or closes the user's personal work.
        """
        self.user_notepad_handles = {w._hWnd for w in gw.getWindowsWithTitle("Notepad")}  # noqa: SLF001
        if self.user_notepad_handles:
            print(
                f"[SAFE] Ignoring {len(self.user_notepad_handles)} existing Notepads.",
            )

    def _cleanup_bot_windows(self) -> None:
        """Close any Notepad instances NOT belonging to the user's initial snapshot."""
        print("[CLEANUP] Sweeping bot-created windows...")

        bot_windows = [
            w
            for w in gw.getWindowsWithTitle("Notepad")
            if w._hWnd not in self.user_notepad_handles  # noqa: SLF001
        ]

        for win in bot_windows:
            try:
                win.restore()
                win.activate()
                win.close()
                # Handle the 'Do you want to save?' prompt by pressing 'No'
                time.sleep(0.5)
                pyautogui.press("n")
                self.logger.info("cleanup_success", handle=win._hWnd)  # noqa: SLF001
            except Exception as e:
                self.logger.error("cleanup_failed", handle=win._hWnd, error=str(e))  # noqa: SLF001, TRY400
                self.monitor.capture_screenshot(f"cleanup_fail_{win._hWnd}")  # noqa: SLF001

    def _capture_workspace_state(self) -> None:
        """Save the current layout of open windows to restore later."""
        self.monitor.capture_screenshot("workspace_before_minimize")
        self.original_workspace_snapshot = [
            {"window": w, "maximized": w.isMaximized}
            for w in gw.getAllWindows()
            if w.title and not w.isMinimized and w.visible
        ]

    def _restore_workspace_state(self) -> None:
        """Restore user windows to their original positions and Z-order."""
        print("[RESTORE] Restoring user workspace...")
        for state in reversed(self.original_workspace_snapshot):
            win = state["window"]
            try:
                if state["maximized"]:
                    win.maximize()
                else:
                    win.restore()
                win.activate()
            except Exception as e:
                self.logger.warning(
                    "restore_item_failed",
                    title=win.title,
                    error=str(e),
                )
        self.monitor.capture_screenshot("workspace_restored")

    # =========================================================
    # INTERNAL UTILITIES
    # =========================================================

    def _robotic_launch_pipeline(self) -> bool:
        """Handle retry logic and environment clearing for app launch.

        Returns:
            True if the application was successfully launched via Strategy.

        """
        for attempt in range(self.MAX_LAUNCH_ATTEMPTS):
            self._reset_desktop_environment()
            start_time = time.perf_counter()

            if self.strategy.launch():
                elapsed = time.perf_counter() - start_time
                self.monitor.log_step(
                    "launch_success",
                    self.state.name,
                    {"attempt": attempt + 1, "duration_sec": round(elapsed, 2)},
                )
                return True

            self.metrics["launch_failures"] += 1
            # Exponential backoff: 1s, 2s, 4s...
            time.sleep(2**attempt)
        return False

    def _get_active_bot_notepad(self) -> gw.Win32Window | None:
        """Locate a visible Notepad window that the bot created."""
        wins = [
            w
            for w in gw.getWindowsWithTitle("Notepad")
            if w.visible
            and not w.isMinimized
            and w._hWnd not in self.user_notepad_handles  # noqa: SLF001
        ]
        return cast("gw.Win32Window", wins[0]) if wins else None

    def _reset_desktop_environment(self) -> None:
        """Minimize windows to provide a 'clean plate' for computer vision."""
        pyautogui.hotkey("win", "m")
        time.sleep(1.5)
        pyautogui.click(1, 1)

    def _transition(self, new_state: TaskState) -> None:
        """Execute a state change and log telemetry for the audit trail."""
        duration = time.perf_counter() - self.state_start_time

        self.logger.info(
            "state_transition",
            from_state=self.state.name,
            to_state=new_state.name,
            duration_sec=round(duration, 2),
        )
        self.monitor.log_step(
            "state_transition",
            new_state.name,
            {"duration_sec": round(duration, 2)},
        )

        self.state = new_state
        self.state_start_time = time.perf_counter()

    # =========================================================
    # CORE LOGIC
    # =========================================================

    def _fetch_posts_from_api(self) -> list[Post]:
        """Fetch post data with integrated retry resilience."""

        def request_action() -> list[Post]:
            print("[INFO] Fetching posts from remote API...")
            response = requests.get(self.api_url, timeout=API_TIMEOUT_SEC)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                msg = "API response must be a JSON list."
                raise TypeError(msg)
            return data

        try:
            return retry(request_action)
        except Exception as e:
            print(f"[ERROR] API data acquisition failed: {e}")
            return []

    def _save_post_to_notepad(self, post: Post) -> bool:
        """Execute the UI automation sequence to save data to a file.

        Note: This uses clipboard operations (Ctrl+C/V) for speed and
        character accuracy over manual typing.
        """
        win = self._get_active_bot_notepad()
        if not win:
            print(f"[ERROR] Bot Notepad not found for post {post['id']}")
            return False

        with contextlib.suppress(Exception):
            win.activate()

        time.sleep(0.5)
        pyautogui.hotkey("ctrl", "n")  # Ensure a fresh document

        content = f"Title: {post['title']}\n\n{post['body']}"
        pyperclip.copy(content)

        # Sequence: Select All -> Paste -> Save
        pyautogui.hotkey("ctrl", "a")
        pyautogui.hotkey("ctrl", "v")
        pyautogui.hotkey("ctrl", "s")

        # Poll for the Windows 'Save As' dialog (up to 5 seconds)
        save_dialog = None
        for _ in range(10):
            dialogs = [w for w in gw.getWindowsWithTitle("Save As") if w.visible]
            if dialogs:
                save_dialog = dialogs[0]
                break
            time.sleep(0.5)

        if not save_dialog:
            self.monitor.capture_screenshot(f"missing_save_dialog_id_{post['id']}")
            return False

        file_path = str(PROJECT_DIR / f"post_{post['id']}.txt")

        # UI Sequence to enter file path
        pyautogui.hotkey("alt", "n")
        pyautogui.hotkey("ctrl", "a")
        pyautogui.press("backspace")
        pyperclip.copy(file_path)
        pyautogui.hotkey("ctrl", "v")
        pyautogui.press("enter")

        # Handle 'Confirm Overwrite' if it appears
        time.sleep(0.7)
        if any(w.visible for w in gw.getWindowsWithTitle("Confirm Save As")):
            pyautogui.press("y")

        pyautogui.hotkey("ctrl", "w")  # Close document

        # Physical file verification
        time.sleep(1.0)
        if not (PROJECT_DIR / f"post_{post['id']}.txt").exists():
            self.logger.error("file_write_verification_failed", post_id=post["id"])
            return False

        with contextlib.suppress(Exception):
            win.close()
        return True

    def run(self) -> None:
        """Execute the main finite state machine loop to process automation tasks."""
        try:
            # Setup Phase
            self._capture_workspace_state()
            self._archive_existing_outputs()
            self._snapshot_user_windows()
            self._cleanup_bot_windows()

            self._transition(TaskState.FETCH_POSTS)
            self.posts = self._fetch_posts_from_api()

            if not self.posts:
                self._transition(TaskState.FAILED)
                return

            self._transition(TaskState.PROCESS_POST)

            # Operational Phase
            while self.state not in {TaskState.DONE, TaskState.FAILED}:
                # Watchdog: Stop execution if it exceeds the global limit
                if (
                    time.perf_counter() - self.total_run_start_time
                    > self.GLOBAL_TIMEOUT_SEC
                ):
                    self.logger.error("watchdog_timeout_triggered")
                    self._transition(TaskState.FAILED)
                    break

                if self.state == TaskState.PROCESS_POST:
                    if self.current_post_index >= min(len(self.posts), MAX_POSTS):
                        self._transition(TaskState.RESTORE)
                    else:
                        print(
                            f"\n[FSM] Processing Post ID: {self.posts[self.current_post_index]['id']}",
                        )
                        self._transition(TaskState.LAUNCH)

                elif self.state == TaskState.LAUNCH:
                    if self._robotic_launch_pipeline():
                        self._save_strategy_artifact()
                        self._transition(TaskState.SAVE_AND_CLOSE)
                    else:
                        self.monitor.capture_screenshot("launch_exhausted_retries")
                        self._transition(TaskState.NEXT_POST)

                elif self.state == TaskState.SAVE_AND_CLOSE:
                    current_post = self.posts[self.current_post_index]
                    if self._save_post_to_notepad(current_post):
                        self.metrics["processed_count"] += 1
                    self._transition(TaskState.NEXT_POST)

                elif self.state == TaskState.NEXT_POST:
                    self.current_post_index += 1
                    self._transition(TaskState.PROCESS_POST)

                elif self.state == TaskState.RESTORE:
                    self._restore_workspace_state()
                    self._transition(TaskState.DONE)

        except Exception:
            self.monitor.capture_screenshot("fatal_pipeline_error")
            self.logger.exception("task_execution_aborted")
            self._transition(TaskState.FAILED)
        finally:
            self._cleanup_bot_windows()
            self._restore_workspace_state()

            runtime = time.perf_counter() - self.total_run_start_time
            self.monitor.finalize(self.metrics)
            print(f"[COMPLETED] Total Runtime: {runtime:.2f}s")

    def _save_strategy_artifact(self) -> None:
        """Extract and save a visual debug frame from the perception engine."""
        debug_frame = self.strategy.get_debug_frame()

        if debug_frame is not None:
            artifact_dir = self.monitor.run_dir / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            save_path = artifact_dir / f"perception_view_{int(time.time())}.png"
            cv2.imwrite(str(save_path), debug_frame)
