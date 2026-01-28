"""Automate Notepad tasks using AI grounding and PyAutoGUI."""

import contextlib
import os
import time
from pathlib import Path
from typing import TypedDict

import pyautogui
import pygetwindow as gw
import pyperclip
import requests
from dotenv import load_dotenv

from llm_solution.grounding_engine import AiGroundingEngine

load_dotenv()

TEXT_QUERY = "Notepad Shortcut"
PROJECT_DIR = Path.home() / "Desktop" / "tjm-project"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)


class Post(TypedDict):
    """Represent the schema for a social media post from the JSONPlaceholder API."""

    userId: int
    id: int
    title: str
    body: str


class NotepadTask:
    """Handle the automation of opening Notepad, writing content, and saving files."""

    def __init__(self) -> None:
        """Initialize the grounding engine and workspace state."""
        self.engine = AiGroundingEngine()
        self.original_windows: list[gw.Win32Window] = []
        posts_api = os.getenv("API_URL")
        if not posts_api:
            msg = "API_URL not found in environment variables."
            raise ValueError(msg)
        self.posts_api = posts_api

    def _capture_workspace_state(self) -> None:
        """Capture currently active and visible windows to restore them later."""
        self.original_windows = [
            w for w in gw.getAllWindows() if w.title and not w.isMinimized and w.visible
        ]

    def _restore_workspace_state(self) -> None:
        """Restore previously captured windows to their original state."""
        print("[RESTORE] Restoring workspace...")
        for w in reversed(self.original_windows):
            with contextlib.suppress(Exception):
                w.restore()
                w.activate()

    def _get_ai_grounding(self, instruction: str) -> tuple[int, int] | None:
        """Resolve pixel coordinates for a UI element by querying the engine."""
        results = self.engine.resolve_coordinates(
            instruction=instruction,
            target_window="Entire Desktop",
            scale_to_pixels=True,
        )

        if not results:
            return None

        # Sort results by score (descending) to get the best match first
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Check if the best match meets our quality threshold
        if results[0].get("score", 0.0) < 0.5:
            print(
                f"[WARN] Best match score ({results[0]['score']}) is below threshold.",
            )
            return None

        coords = results[0]["coords"]
        return (coords[0], coords[1])

    def _get_active_notepad(self) -> gw.Win32Window | None:
        """Retrieve the currently active and visible Notepad window."""
        valid = [
            w
            for w in gw.getWindowsWithTitle("Notepad")
            if w.visible and not w.isMinimized
        ]
        return valid[0] if valid else None

    def launch_by_grounding(self) -> bool:
        """Minimize windows and use AI to double-click the Notepad icon."""
        pyautogui.hotkey("win", "m")
        time.sleep(1.5)
        pyautogui.click(1, 1)  # Clear focus/hovers

        target = self._get_ai_grounding(f"The {TEXT_QUERY} desktop icon")
        if not target:
            print(f"[ERROR] Grounding failed: No matches for '{TEXT_QUERY}'.")
            return False

        cx, cy = target
        pyautogui.doubleClick(cx, cy)

        # Verification loop
        for _ in range(12):
            if self._get_active_notepad():
                return True
            time.sleep(0.5)
        return False

    def save_and_close(self, post: Post) -> None:
        """Type content, save file, and close the application/tab."""
        win = self._get_active_notepad()
        if not win:
            return

        with contextlib.suppress(Exception):
            win.activate()
        time.sleep(0.5)

        # 1. UNIVERSAL NEW DOCUMENT (Handles Win 11 tabs)
        pyautogui.hotkey("ctrl", "n")
        time.sleep(0.5)

        # 2. INPUT DATA
        content = f"Title: {post['title']}\n\n{post['body']}"
        pyperclip.copy(content)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.5)

        # 3. TRIGGER AND WAIT FOR SAVE DIALOG
        pyautogui.hotkey("ctrl", "s")

        save_dialog = None
        for _ in range(10):
            dialogs = [w for w in gw.getWindowsWithTitle("Save As") if w.visible]
            if dialogs:
                save_dialog = dialogs[0]
                break
            time.sleep(0.5)

        if save_dialog:
            # 4. BOMB-PROOF FILE ENTRY
            file_path = str(PROJECT_DIR / f"post_{post['id']}.txt")

            pyautogui.hotkey("alt", "n")  # Focus file name field
            time.sleep(0.2)
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")

            pyperclip.copy(file_path)
            pyautogui.hotkey("ctrl", "v")
            pyautogui.press("enter")
            time.sleep(1.0)

            # 5. HANDLE OVERWRITE
            confirms = [
                w for w in gw.getWindowsWithTitle("Confirm Save As") if w.visible
            ]
            if confirms:
                pyautogui.press("y")
                time.sleep(0.5)

            # 6. UNIVERSAL CLEANUP (Close tab then window)
            pyautogui.hotkey("ctrl", "w")
            time.sleep(0.3)
            with contextlib.suppress(Exception):
                if win.visible:
                    win.close()

            print(f"[COMPLETE] Processed post {post['id']}.")
        else:
            print(f"[ERROR] Save dialog failed to appear for post {post['id']}")

    def run(self) -> None:
        """Execute the automation loop for the first 10 API posts."""
        self._capture_workspace_state()

        try:
            response = requests.get(self.posts_api, timeout=10)
            posts: list[Post] = response.json()

            for post in posts[0:10]:
                print(f"\n[STEP] Post {post['id']}")
                if self.launch_by_grounding():
                    self.save_and_close(post)
                else:
                    print("[FATAL] Skipping post due to grounding failure.")
                time.sleep(1.0)
        finally:
            self.engine.cleanup()
            self._restore_workspace_state()
            if Path("raw_capture.png").exists():
                Path("raw_capture.png").unlink()


def run() -> None:
    """Start the Notepad automation task."""
    NotepadTask().run()
