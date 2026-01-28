"""Service for capturing desktop and application screenshots with state restoration."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pyautogui
import pygetwindow as gw
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


class ScreenshotService:
    """Handle desktop and app capture with surgical state management."""

    def __init__(self, settle_delay: float = 1.0) -> None:
        """Initialize service with settle delay."""
        self._settle_delay = settle_delay

    def capture_desktop(self, output_path: Path) -> Path:
        """Capture the full desktop after clearing the workspace.

        Workflow:
            1. Minimize all windows via Win+D.
            2. Clear mouse hovers by clicking the corner.
            3. Capture and save full screen.
            4. Restore original window state.
        """
        self.log("[PROCESS] Clearing workspace for desktop capture.")
        self._toggle_desktop()

        try:
            time.sleep(self._settle_delay)
            pyautogui.click(1, 1)
            image = pyautogui.screenshot()
            image.save(output_path)
        finally:
            self._toggle_desktop()
            time.sleep(0.5)

        return output_path

    def capture_app_window(
        self,
        window_title: str,
        output_path: Path,
    ) -> tuple[Path, tuple[int, int, int, int]]:
        """Isolate a specific application window against a blanked-out background.

        This method ensures the AI sees only the target application by masking the
        background and taskbar with a solid black fill.

        Workflow:
            1. State Snapshot: Record current visibility of all open windows.
            2. Clear Stage: Minimize all windows to prevent background overlap.
            3. Target Isolation: Restore and focus only the target window.
            4. Capture & Mask: Capture full screen, then black out all pixels
               outside the target window's bounds (including the taskbar).
            5. Restoration: Return all windows to their original positions/states.
        """
        # 1. State Snapshot
        all_wins = gw.getAllWindows()
        window_snapshot = [
            {"win": w, "was_max": w.isMaximized, "was_min": w.isMinimized}
            for w in all_wins
            if w.title.strip() and w.visible
        ]

        try:
            # 2. Clear Stage
            self._toggle_desktop()
            time.sleep(0.5)

            # 3. Target Isolation
            windows = gw.getWindowsWithTitle(window_title)
            if not windows:
                msg = f"Window not found: {window_title}"
                raise ValueError(msg)

            target_win = windows[0]
            target_win.restore()
            target_win.activate()
            time.sleep(self._settle_delay)

            # 4. Capture & Mask
            full_ss = pyautogui.screenshot()

            # Create the "Blank Background" mask
            mask = Image.new("RGB", full_ss.size, (0, 0, 0))  # Solid Black
            rect = (
                target_win.left,
                target_win.top,
                target_win.left + target_win.width,
                target_win.top + target_win.height,
            )

            # Paste the app pixels onto the black background
            app_region = full_ss.crop(rect)
            mask.paste(app_region, (target_win.left, target_win.top))
            mask.save(output_path)

        finally:
            # 5. Surgical Restoration
            self.log("[PROCESS] Restoring original window environment.")
            for item in reversed(window_snapshot):
                win = item["win"]
                if item["was_max"]:
                    win.maximize()
                elif not item["was_min"]:
                    win.restore()
            time.sleep(0.5)

        return output_path, (
            target_win.left,
            target_win.top,
            target_win.width,
            target_win.height,
        )

    def _toggle_desktop(self) -> None:
        """Toggle Windows 'Show Desktop' (Win+D)."""
        pyautogui.hotkey("win", "d")

    def log(self, msg: str) -> None:
        """Log the service message."""
        print(msg)
