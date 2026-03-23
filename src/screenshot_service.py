"""Provide surgical application and desktop screen capture services.

Manage the desktop state to ensure that AI and Computer Vision grounding
engines receive clean, uncluttered visual input without background noise.
"""

from __future__ import annotations

import time

import numpy as np
import pyautogui
import pygetwindow as gw
from PIL import Image


class ScreenshotService:
    """Service for capturing desktop and application windows with UI state management.

    Provides:
      - Workspace snapshot and restoration.
      - Full desktop or app window screenshots.
      - Waiting for UI animations to settle.

    Dependencies:
        pyautogui, pygetwindow, PIL (Pillow), numpy

    Attributes:
        _settle_delay (float): Delay in seconds for UI animations to finish.
        _window_snapshot (list[dict]): Stored state of windows for restoration.

    """

    def __init__(self, settle_delay: float = 0.3) -> None:
        """Initialize the screenshot service."""
        self._settle_delay = settle_delay
        self._window_snapshot: list[dict] = []

    def snapshot_workspace(self) -> None:
        """Capture the current state of all visible windows for restoration later."""
        # Get all windows currently open
        all_wins = gw.getAllWindows()
        # Store window references and their maximized/minimized state
        self._window_snapshot = [
            {"win": w, "was_max": w.isMaximized, "was_min": w.isMinimized}
            for w in all_wins
            if w.title.strip() and w.visible  # ignore untitled or invisible windows
        ]
        self.log(
            f"[INIT] Workspace snapshot taken: {len(self._window_snapshot)} windows.",
        )

    def restore_workspace(self) -> None:
        """Restore all windows to their original states in reverse Z-order."""
        if not self._window_snapshot:
            self.log("[WARN] No workspace snapshot found to restore.")
            return

        self.log("[PROCESS] Restoring original window environment.")
        for item in reversed(self._window_snapshot):
            win = item["win"]
            try:
                if item["was_max"]:
                    win.maximize()  # Restore maximized windows
                elif not item["was_min"]:
                    win.restore()  # Restore normal windows
            except Exception as e:
                self.log(f"[WARN] Could not restore window '{win.title}': {e}")
        # Give Windows a moment to finish rendering restored windows
        time.sleep(0.5)

    def capture_desktop(self) -> Image.Image:
        """Capture a screenshot of the full desktop after clearing the workspace."""
        # Minimize all windows to isolate desktop
        self.toggle_desktop()
        # Wait for UI animations to settle
        time.sleep(self._settle_delay)
        # Click on top-left corner to ensure no window is focused
        pyautogui.click(1, 1)
        # Return the full desktop screenshot
        return pyautogui.screenshot()

    def capture_app_window(
        self,
        window_title: str,
        *,
        minimize_others: bool = True,
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        """Capture a screenshot of a specific application window.

        Optionally minimizes all other windows to reduce background noise
        and ensure the target window is fully visible.

        Args:
            window_title: The exact title of the window to capture.
            minimize_others: If True, minimizes all other windows before capturing.

        Returns:
            A tuple containing:
                - mask: A PIL.Image where only the target window is visible.
                - rect: The coordinates (left, top, width, height) of the target window.

        Raises:
            ValueError: If no window matches the given title.

        """
        # Step 1: Find the target window
        windows = gw.getWindowsWithTitle(window_title)
        if not windows:
            msg = f"Window not found: {window_title}"
            raise ValueError(msg)
        target_win = windows[0]

        # Step 2: Minimize other windows if requested
        if minimize_others:
            self.toggle_desktop()
            time.sleep(self._settle_delay)

        # Step 3: Restore and activate only the target window
        target_win.restore()
        target_win.activate()

        # Step 4: Wait until UI finishes animating
        self._wait_for_ui_settle()

        # Step 5: Capture full desktop screenshot
        full_ss = pyautogui.screenshot()

        # Step 6: Mask the desktop so only the target window is visible
        mask = Image.new("RGB", full_ss.size, (0, 0, 0))
        crop_rect = (
            target_win.left,
            target_win.top,
            target_win.left + target_win.width,
            target_win.top + target_win.height,
        )
        mask.paste(full_ss.crop(crop_rect), (target_win.left, target_win.top))

        # Step 7: Return masked image and window coordinates
        rect = (target_win.left, target_win.top, target_win.width, target_win.height)
        return mask, rect

    def toggle_desktop(self) -> None:
        """Minimize all windows using Windows 'Show Desktop' shortcut (Win+M)."""
        pyautogui.hotkey("win", "m")  # Side effect: all windows minimized

    def log(self, msg: str) -> None:
        """Print a log message to the console."""
        print(msg)  # Debug / workflow tracing

    def _wait_for_ui_settle(self, timeout: float = 2.0) -> None:
        """Wait until screen pixels stop changing (UI animations finished)."""
        start_time = time.perf_counter()
        last_frame = np.array(pyautogui.screenshot())
        while time.perf_counter() - start_time < timeout:
            time.sleep(0.2)  # Polling interval
            current_frame = np.array(pyautogui.screenshot())
            if np.array_equal(last_frame, current_frame):
                break  # Stop waiting if screen is stable
            last_frame = current_frame
