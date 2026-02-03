"""Automate Notepad tasks using computer vision grounding."""

import os
import time
from contextlib import suppress
from pathlib import Path
from typing import TypedDict

import cv2
import pyautogui
import pygetwindow as gw
import pyperclip
import requests
from dotenv import load_dotenv

from opencv_solution.grounding_engine import DesktopGroundingEngine
from screenshot_service import ScreenshotService

# CONFIG
ICON_PATH = Path("notepad_icon.png")
TEXT_QUERY = "Notepad"
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
PROJECT_DIR = Path.home() / "Desktop" / "tjm-project"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()


class Post(TypedDict):
    """Represent the schema for a social media post from the JSONPlaceholder API."""

    userId: int
    id: int
    title: str
    body: str


class NotepadTask:
    """Handle automated interaction with Notepad via grounding and GUI automation."""

    def __init__(self) -> None:
        """Initialize the grounding engine and workspace state."""
        self.engine = DesktopGroundingEngine(tesseract_path=TESS_PATH)
        self.ss = ScreenshotService()
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
            with suppress(Exception):
                w.restore()
                w.activate()

    def _get_active_notepad(self) -> gw.Win32Window | None:
        """Find the Notepad window that is actually visible."""
        valid = [
            w
            for w in gw.getWindowsWithTitle("Notepad")
            if w.visible and not w.isMinimized
        ]
        return valid[0] if valid else None

    def _validate_posts(self, posts: object) -> list[Post]:
        if not isinstance(posts, list):
            msg = "Unexpected API response format, expected list"
            raise TypeError(msg)
        return posts

    def launch_by_grounding(self) -> bool:
        """Force a desktop scan and open Notepad via icon/text detection."""
        # 1. Minimize everything to show the desktop
        pyautogui.hotkey("win", "m")
        time.sleep(1.5)

        # 2. Clear hover states
        pyautogui.click(1, 1)
        time.sleep(0.2)

        # 3. Capture desktop
        temp_ss = Path("grounding_temp.png")
        pyautogui.screenshot(str(temp_ss))

        # 4. Locate all Candidates
        engine_config = {
            "use_ocr": True,
            "use_color": True,
            "use_multiscale": True,
            "num_cores": 4,
        }

        raw_candidates = self.engine.locate_elements(
            screenshot_path=temp_ss,
            icon_image=ICON_PATH,
            text_query=TEXT_QUERY,
            threshold=0.5,
            config=engine_config,
        )

        # 5. Select the best candidate
        best_cand = self.engine.select_best_candidate(raw_candidates, priority="fusion")

        if not best_cand:
            print(f"[ERROR] Grounding failed: No matches for '{TEXT_QUERY}'.")
            return False

        # 6. Attempt to open
        print(
            f"[TARGET] Targeting {best_cand.method} at ({best_cand.x}, {best_cand.y})",
        )
        print(
            f"[SCORE] Image: {best_cand.img_score:.2f} | "
            f"Text: {best_cand.txt_score:.2f}",
        )

        screen_w, screen_h = pyautogui.size()
        img = cv2.imread(str(temp_ss))
        if img is None:
            print("[ERROR] Could not read screenshot file.")
            return False

        img_h, img_w = img.shape[:2]
        scale_x = screen_w / img_w
        scale_y = screen_h / img_h

        screen_x = int(best_cand.x * scale_x)
        screen_y = int(best_cand.y * scale_y)

        # bounds check
        if not (0 <= screen_x < screen_w and 0 <= screen_y < screen_h):
            print(
                f"[ERROR] Candidate coords out of screen bounds: ({screen_x},{screen_y})",
            )
            return False

        pyautogui.doubleClick(screen_x, screen_y)

        # Verification loop
        for _ in range(12):
            if self._get_active_notepad():
                print("[SUCCESS] Notepad Window Detected.")
                return True
            time.sleep(0.5)

        return False

    def save_and_close(self, post: Post) -> None:
        """Type content, save file, and close the application.

        Provide compatibility with Windows 10 (windows) and Windows 11 (tabs).
        """
        win = self._get_active_notepad()
        if not win:
            print(f"[ERROR] No active Notepad window for post {post['id']}.")
            return

        with suppress(Exception):
            win.activate()
        time.sleep(0.5)

        # 1. UNIVERSAL NEW DOCUMENT
        pyautogui.hotkey("ctrl", "n")
        time.sleep(0.5)

        # 2. INPUT DATA
        content = f"Title: {post['title']}\n\n{post['body']}"
        pyperclip.copy(content)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.3)

        # 3. TRIGGER SAVE DIALOG
        pyautogui.hotkey("ctrl", "s")

        # Search for the 'Save As' window
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

            pyautogui.hotkey("alt", "n")
            time.sleep(0.2)
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")

            pyperclip.copy(file_path)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.3)
            pyautogui.press("enter")

            # 5. HANDLE OVERWRITE
            time.sleep(0.7)
            confirm = [
                w for w in gw.getWindowsWithTitle("Confirm Save As") if w.visible
            ]
            if confirm:
                pyautogui.press("y")
                time.sleep(0.3)

            # 6. UNIVERSAL CLEANUP
            pyautogui.hotkey("ctrl", "w")
            time.sleep(0.3)

            with suppress(Exception):
                if win.visible:
                    win.close()

            print(f"[COMPLETE] Processed post {post['id']} and cleaned up.")
        else:
            print(f"[WARNING] Save dialog failed to appear for post {post['id']}.")

    def run(self) -> None:
        """Execute the full grounding and automation pipeline."""
        self._capture_workspace_state()

        try:
            response = requests.get(self.posts_api, timeout=10)
            response.raise_for_status()
            raw_posts = response.json()

            posts = self._validate_posts(raw_posts)

        except requests.RequestException as e:
            print(f"[ERROR] Network/API failure: {e}")

        except TypeError as e:
            print(f"[ERROR] Invalid API payload: {e}")

        try:
            for post in posts[:10]:
                print(f"\n[STEP] Post {post['id']}")

                if self.launch_by_grounding():
                    self.save_and_close(post)
                else:
                    print("[FATAL] Skipping post due to grounding failure.")

                time.sleep(1.0)

        finally:
            self._restore_workspace_state()
            temp_file = Path("grounding_temp.png")
            if temp_file.exists():
                temp_file.unlink()
            print("\n[INFO] All steps completed.")


def run() -> None:
    """Run the Notepad automation task."""
    NotepadTask().run()
