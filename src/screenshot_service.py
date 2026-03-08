"""Provide surgical application and desktop screen capture services.

Manage the desktop state to ensure that AI and Computer Vision grounding
engines receive clean, uncluttered visual input without background noise.
"""

from __future__ import annotations

import time

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

    def capture_desktop(self) -> Image.Image:
        """Capture the full desktop after minimizing all active windows.

        Returns:
            The PIL Image object of the cleared desktop screenshot.

        """
        self.log("[PROCESS] Clearing workspace for desktop capture.")
        self._toggle_desktop()

        try:
            # Allow time for 'Minimize All' animations to complete
            time.sleep(self._settle_delay)

            # Move mouse to corner to avoid hovering over UI elements/tooltips
            pyautogui.click(1, 1)

            # Return the screenshot as a PIL Image object
            return pyautogui.screenshot()

        finally:
            # Restore the windows to their original state
            self._toggle_desktop()
            time.sleep(0.5)

    def capture_app_window(
        self,
        window_title: str,
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        """Isolate a specific application window against a blacked-out background.

        Mask the background and taskbar with a solid black fill in-memory to
        ensure the grounding engine sees only the target application.

        Args:
            window_title: The title of the window to isolate.

        Returns:
            A tuple containing the masked PIL Image and the (x, y, w, h) bounds.

        Raises:
            ValueError: If no window with the specified title is found.

        """
        # 1. Snapshot the current visibility and state of all open windows
        all_wins = gw.getAllWindows()
        window_snapshot = [
            {"win": w, "was_max": w.isMaximized, "was_min": w.isMinimized}
            for w in all_wins
            if w.title.strip() and w.visible
        ]

        try:
            # 2. Minimize everything to prevent background noise
            self._toggle_desktop()
            time.sleep(0.5)

            # 3. Locate and focus the target application
            windows = gw.getWindowsWithTitle(window_title)
            if not windows:
                msg = f"Window not found: {window_title}"
                raise ValueError(msg)

            target_win = windows[0]
            target_win.restore()
            target_win.activate()
            time.sleep(self._settle_delay)

            # 4. Perform the surgical crop and mask
            full_ss = pyautogui.screenshot()

            mask = Image.new("RGB", full_ss.size, (0, 0, 0))

            # Calculate coordinates for the crop (left, top, right, bottom)
            crop_rect = (
                target_win.left,
                target_win.top,
                target_win.left + target_win.width,
                target_win.top + target_win.height,
            )

            # Extract the app region and paste it onto the black canvas
            app_region = full_ss.crop(crop_rect)
            mask.paste(app_region, (target_win.left, target_win.top))

            # Store metadata for the grounding engine: (x, y, w, h)
            rect = (
                target_win.left,
                target_win.top,
                target_win.width,
                target_win.height,
            )

            return mask, rect

        finally:
            # 5. Restore original window environment in reverse Z-order
            self.log("[PROCESS] Restoring original window environment.")
            for item in reversed(window_snapshot):
                win = item["win"]
                try:
                    if item["was_max"]:
                        win.maximize()
                    elif not item["was_min"]:
                        win.restore()
                except Exception as e:
                    self.log(f"[WARN] Could not restore window '{win.title}': {e}")
                    continue
            time.sleep(0.5)

    def _toggle_desktop(self) -> None:
        """Toggle the Windows 'Show Desktop' state via Hotkey."""
        pyautogui.hotkey("win", "d")

    def log(self, msg: str) -> None:
        """Log service status to the console."""
        print(msg)
