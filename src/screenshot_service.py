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
    """Manage desktop and app capture with surgical state management."""

    def __init__(self, settle_delay: float = 1.0) -> None:
        """Initialize the service with a custom settle delay.

        Args:
            settle_delay: Wait time in seconds for UI animations to finish.

        """
        self._settle_delay = settle_delay
        self._window_snapshot: list[dict] = []  # Tracks workspace state

    def snapshot_workspace(self) -> None:
        """Capture the current state of all visible windows for later restoration."""
        all_wins = gw.getAllWindows()
        self._window_snapshot = [
            {"win": w, "was_max": w.isMaximized, "was_min": w.isMinimized}
            for w in all_wins
            if w.title.strip() and w.visible
        ]
        self.log(
            f"[INIT] Workspace snapshot taken: {len(self._window_snapshot)} windows.",
        )

    def restore_workspace(self) -> None:
        """Restore windows to their original state in reverse Z-order."""
        if not self._window_snapshot:
            self.log("[WARN] No workspace snapshot found to restore.")
            return

        self.log("[PROCESS] Restoring original window environment.")
        for item in reversed(self._window_snapshot):
            win = item["win"]
            try:
                if item["was_max"]:
                    win.maximize()
                elif not item["was_min"]:
                    win.restore()
            except Exception as e:
                # Fixed Ruff S112 by logging the exception instead of just continuing
                self.log(f"[WARN] Could not restore window '{win.title}': {e}")
                continue
        time.sleep(0.5)

    def capture_desktop(self) -> Image.Image:
        """Capture the full desktop after clearing workspace."""
        self.toggle_desktop()
        time.sleep(self._settle_delay)
        pyautogui.click(1, 1)
        return pyautogui.screenshot()

    def capture_app_window(
        self,
        window_title: str,
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        """Isolate a specific application window against a blacked-out background."""
        windows = gw.getWindowsWithTitle(window_title)
        if not windows:
            msg = f"Window not found: {window_title}"
            raise ValueError(msg)

        target_win = windows[0]
        target_win.restore()
        target_win.activate()
        self.wait_for_ui_settle()

        full_ss = pyautogui.screenshot()
        mask = Image.new("RGB", full_ss.size, (0, 0, 0))

        crop_rect = (
            target_win.left,
            target_win.top,
            target_win.left + target_win.width,
            target_win.top + target_win.height,
        )

        app_region = full_ss.crop(crop_rect)
        mask.paste(app_region, (target_win.left, target_win.top))

        rect = (target_win.left, target_win.top, target_win.width, target_win.height)
        return mask, rect

    def toggle_desktop(self) -> None:
        """Toggle the Windows 'Show Desktop' state via Win+M."""
        pyautogui.hotkey("win", "m")

    def log(self, msg: str) -> None:
        """Log service status to the console."""
        print(msg)

    def wait_for_ui_settle(self, timeout: float = 2.0) -> None:
        """Wait until screen pixels stop changing (animations finished)."""
        start_time = time.perf_counter()
        last_frame = np.array(pyautogui.screenshot())
        while time.perf_counter() - start_time < timeout:
            time.sleep(0.2)
            current_frame = np.array(pyautogui.screenshot())
            if np.array_equal(last_frame, current_frame):
                break
            last_frame = current_frame
