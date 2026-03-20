"""Provide application-specific UI drivers (Page Object Model for Desktop).

Abstracts low-level keystrokes and UI interactions away from the orchestrator.
"""

from __future__ import annotations

import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import pyautogui
import pygetwindow as gw
import pyperclip

from utils import wait_for_window_to_disappear

if TYPE_CHECKING:
    from core import StructuredLogger


class NotepadDriver:
    """Driver for interacting with the Notepad application."""

    def __init__(self, hwnd: int, logger: StructuredLogger) -> None:
        """Initialize the driver bound to a specific Notepad window handle.

        Args:
            hwnd: The OS-level window handle integer.
            logger: The telemetry emitter for structured logging.

        """
        self.hwnd = hwnd
        self.logger = logger

    @property
    def window(self) -> gw.Win32Window | None:
        """Fetch the active pygetwindow object based on the bound HWND."""
        wins = [w for w in gw.getAllWindows() if w._hWnd == self.hwnd]  # noqa: SLF001
        return wins[0] if wins else None

    def focus(self) -> bool:
        """Bring the bound Notepad window to the foreground."""
        win = self.window
        if not win or not win.visible:
            self.logger.error("focus_failed_window_missing", hwnd=self.hwnd)
            return False
        with suppress(Exception):
            win.activate()
        time.sleep(0.5)  # Short settle for OS focus animation
        return True

    def write_content(self, content: str) -> None:
        """Write text into the editor using the clipboard for speed."""
        if not self.focus():
            return

        self.logger.debug("writing_content_started", length=len(content))

        # Windows 11 Notepad handling: Ensure a clean tab
        pyautogui.hotkey("ctrl", "n")
        time.sleep(0.2)

        pyperclip.copy(content)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.hotkey("ctrl", "v")
        self.logger.info("content_injected_via_clipboard")

    def save_as(self, file_path: str) -> bool:
        """Navigate the Save As dialog and verify the file exists on disk."""
        if not self.focus():
            return False

        self.logger.info("save_sequence_started", path=file_path)
        pyautogui.hotkey("ctrl", "s")

        # 1. Wait for Save Dialog
        save_dialog = None
        for _ in range(10):
            dialogs = [w for w in gw.getWindowsWithTitle("Save As") if w.visible]
            if dialogs:
                save_dialog = dialogs[0]
                break
            time.sleep(0.5)

        if not save_dialog:
            self.logger.error("save_dialog_timeout", timeout_sec=5.0)
            return False

        # 2. Input path and confirm
        pyautogui.hotkey("alt", "n")  # Focus filename field
        pyautogui.hotkey("ctrl", "a")
        pyautogui.press("backspace")

        pyperclip.copy(file_path)
        pyautogui.hotkey("ctrl", "v")
        pyautogui.press("enter")

        self.logger.debug("path_submitted_to_dialog", path=file_path)

        # 3. Handle 'Confirm Overwrite' (The "Simple Stuff")
        for _ in range(5):
            confirm = [
                w for w in gw.getWindowsWithTitle("Confirm Save As") if w.visible
            ]
            if confirm:
                self.logger.warning("overwrite_collision_detected", path=file_path)
                pyautogui.press("y")
                break
            time.sleep(0.4)

        # 4. Final verification: UI disappearance + Filesystem check
        dialog_closed = wait_for_window_to_disappear("Save As", timeout=3.0)
        file_exists = Path(file_path).exists()

        if dialog_closed and file_exists:
            self.logger.info("save_successful_and_verified", path=file_path)
            return True

        self.logger.error(
            "save_verification_failed",
            dialog_closed=dialog_closed,
            file_exists=file_exists,
        )
        return False

    def close(self) -> None:
        """Close the Notepad tab and window gracefully."""
        win = self.window
        if not win:
            return

        self.focus()
        self.logger.debug("closing_window", hwnd=self.hwnd)

        # Close tab first (essential for Win11 session state)
        pyautogui.hotkey("ctrl", "w")
        time.sleep(0.3)

        with suppress(Exception):
            win.close()
            time.sleep(0.5)

            # Catch "Do you want to save?" prompts for unexpected changes
            save_prompts = [
                w for w in gw.getWindowsWithTitle("Notepad") if "Save" in w.title
            ]
            if save_prompts:
                self.logger.warning("unsaved_changes_discarded_on_close")
                pyautogui.press("n")
