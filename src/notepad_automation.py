"""Automate Notepad tasks using AI or OpenCV grounding in a DRY pipeline."""

import contextlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import cv2
import pyautogui
import pygetwindow as gw
import pyperclip
import requests
from dotenv import load_dotenv

from llm_solution.grounding_engine import AiGroundingEngine
from opencv_solution.grounding_engine import DesktopGroundingEngine
from screenshot_service import ScreenshotService

if TYPE_CHECKING:
    from collections.abc import Callable

load_dotenv()

TEXT_QUERY = "Notepad Shortcut"
PROJECT_DIR = Path.home() / "Desktop" / "tjm-project"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

# Optional OpenCV configs
ICON_PATH = Path("notepad_icon.png")
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class Post(TypedDict):
    """Represent the schema for a social media post from the JSONPlaceholder API."""

    userId: int
    id: int
    title: str
    body: str


class NotepadTask:
    """Handle automated interaction with Notepad via AI or CV grounding."""

    def __init__(self) -> None:
        """Initialize both grounding engines, screenshot service, and workspace state."""
        self.llm_engine = AiGroundingEngine()
        self.cv_engine = DesktopGroundingEngine(tesseract_path=TESS_PATH)
        self.ss = ScreenshotService()
        self.original_windows: list[gw.Win32Window] = []

        posts_api = os.getenv("API_URL")
        if not posts_api:
            msg = "API_URL not found in environment variables."
            raise ValueError(msg)
        self.posts_api = posts_api

    # -------------------
    # Workspace helpers
    # -------------------
    def _capture_workspace_state(self) -> None:
        self.original_windows = [
            w for w in gw.getAllWindows() if w.title and not w.isMinimized and w.visible
        ]

    def _restore_workspace_state(self) -> None:
        print("[RESTORE] Restoring workspace...")
        for w in reversed(self.original_windows):
            with contextlib.suppress(Exception):
                w.restore()
                w.activate()

    def _get_active_notepad(self) -> gw.Win32Window | None:
        valid = [
            w
            for w in gw.getWindowsWithTitle("Notepad")
            if w.visible and not w.isMinimized
        ]
        return valid[0] if valid else None

    def _fetch_posts(self) -> list[Post]:
        """Fetch post data from the API with 3 retries and exponential backoff."""
        for attempt in range(3):
            try:
                print(f"[INFO] Fetching posts (Attempt {attempt + 1}/3)...")
                response = requests.get(self.posts_api, timeout=10)
                response.raise_for_status()
                return self._validate_posts(response.json())
            except (requests.RequestException, TypeError) as e:
                if attempt == 2:
                    print(f"[ERROR] API failed after 3 attempts: {e}")
                    return []
                time.sleep(2**attempt)
        return []

    def _validate_posts(self, posts: object) -> list[Post]:
        if not isinstance(posts, list):
            msg = "Unexpected API response format, expected list"
            raise TypeError(msg)
        return posts

    # -------------------
    # Core automation loop
    # -------------------
    def automation_loop(self, launch_func: Callable[[], bool]) -> None:
        """Run a generic automation loop for the first 10 posts using the provided launch function."""
        self._capture_workspace_state()

        try:
            posts = self._fetch_posts()
            if not posts:
                return

            for post in posts[:10]:
                print(f"\n[STEP] Post {post['id']}")
                if launch_func():
                    self.save_and_close(post)
                else:
                    print("[FATAL] Skipping post due to grounding failure.")
                time.sleep(1.0)
        finally:
            self._restore_workspace_state()
            temp_files = [
                "raw_capture.png",
                "grounding_temp.png",
                "ai_vision_input.png",
                "capture_state.png",
            ]
            for f in temp_files:
                Path(f).unlink(missing_ok=True)

            print("\n[INFO] All steps completed.")

    # -------------------
    # Save & close logic
    # -------------------
    def save_and_close(self, post: Post) -> None:
        """Type content, save file, and close the Notepad window."""
        win = self._get_active_notepad()
        if not win:
            print(f"[ERROR] No active Notepad window for post {post['id']}.")
            return

        with contextlib.suppress(Exception):
            win.activate()
        time.sleep(0.5)

        pyautogui.hotkey("ctrl", "n")  # New document
        time.sleep(0.5)

        content = f"Title: {post['title']}\n\n{post['body']}"
        pyperclip.copy(content)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.3)

        pyautogui.hotkey("ctrl", "s")  # Trigger Save

        # Wait for Save As dialog
        save_dialog = None
        for _ in range(10):
            dialogs = [w for w in gw.getWindowsWithTitle("Save As") if w.visible]
            if dialogs:
                save_dialog = dialogs[0]
                break
            time.sleep(0.5)

        if save_dialog:
            file_path = str(PROJECT_DIR / f"post_{post['id']}.txt")
            pyautogui.hotkey("alt", "n")
            time.sleep(0.2)
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")
            pyperclip.copy(file_path)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.3)
            pyautogui.press("enter")

            time.sleep(0.7)
            confirm = [
                w for w in gw.getWindowsWithTitle("Confirm Save As") if w.visible
            ]
            if confirm:
                pyautogui.press("y")
                time.sleep(0.3)

            pyautogui.hotkey("ctrl", "w")
            time.sleep(0.3)
            with contextlib.suppress(Exception):
                if win.visible:
                    win.close()
            print(f"[COMPLETE] Processed post {post['id']} and cleaned up.")
        else:
            print(f"[WARNING] Save dialog failed for post {post['id']}.")

    # -------------------
    # LLM & OpenCV launchers
    # -------------------
    def launch_llm(self) -> bool:
        """Use AI grounding engine to locate Notepad icon and double-click it."""
        pyautogui.hotkey("win", "m")
        time.sleep(1.5)
        pyautogui.click(1, 1)

        def engine_logger(msg: str) -> None:
            level = "INFO"
            if "[SUCCESS]" in msg:
                level = "SUCCESS"
            elif "[ERROR]" in msg:
                level = "ERROR"
            clean_msg = (
                msg.replace("[INFO]", "")
                .replace("[SUCCESS]", "")
                .replace("[ERROR]", "")
                .strip()
            )
            print(f"[ENGINE][{level}] {clean_msg}")

        results = self.llm_engine.resolve_coordinates(
            instruction=f"The {TEXT_QUERY} desktop icon",
            target_window="Entire Desktop",
            reference_image_path=None,
            scale_to_pixels=True,
            verify_after_action=False,
            logger_callback=engine_logger,
        )

        if not results:
            print(f"[ERROR] No matches returned by AI for '{TEXT_QUERY}'.")
            return False

        # Sort results by best score first
        results.sort(key=lambda x: (-x.get("score", 0.0), x.get("rank", 1)))

        for node in results:
            coords = node.get("coords", [0, 0])
            score = node.get("score", 0.0)

            if score >= 0.5:
                print(
                    f"[INFO] Attempting launch with AI candidate at {coords} (Score: {score})",
                )
                pyautogui.doubleClick(coords[0], coords[1])

                # Verification loop for THIS specific candidate
                for _ in range(6):
                    if self._get_active_notepad():
                        print("[SUCCESS] Notepad launched via AI candidate.")
                        return True
                    time.sleep(0.5)

                print(
                    f"[WARN] Candidate at {coords} failed to launch Notepad. Trying next...",
                )

        print(f"[ERROR] All {len(results)} AI candidates failed.")
        return False

    def launch_opencv(self) -> bool:
        """Use OpenCV grounding engine to locate Notepad icon and double-click it."""
        pyautogui.hotkey("win", "m")
        time.sleep(1.5)
        pyautogui.click(1, 1)
        temp_ss = Path("grounding_temp.png")
        pyautogui.screenshot(str(temp_ss))

        engine_config = {
            "use_ocr": True,
            "use_color": True,
            "use_multiscale": True,
            "num_cores": 8,
        }
        raw_candidates = self.cv_engine.locate_elements(
            screenshot_path=temp_ss,
            icon_image=ICON_PATH,
            text_query=TEXT_QUERY,
            threshold=0.5,
            config=engine_config,
        )

        raw_candidates.sort(key=lambda x: x.score, reverse=True)

        screen_w, screen_h = pyautogui.size()
        img = cv2.imread(str(temp_ss))
        if img is None:
            print("[ERROR] Could not read screenshot file.")
            return False

        img_h, img_w = img.shape[:2]
        scale_x, scale_y = screen_w / img_w, screen_h / img_h

        for cand in raw_candidates:
            screen_x = int(cand.x * scale_x)
            screen_y = int(cand.y * scale_y)

            if 0 <= screen_x < screen_w and 0 <= screen_y < screen_h:
                print(
                    f"[INFO] Attempting launch with CV candidate at ({screen_x}, {screen_y}) (Score: {cand.score})",
                )
                pyautogui.doubleClick(screen_x, screen_y)

                # Verification loop for THIS specific candidate
                for _ in range(6):
                    if self._get_active_notepad():
                        print("[SUCCESS] Notepad launched via OpenCV candidate.")
                        return True
                    time.sleep(0.5)

                print(
                    f"[WARN] CV candidate at ({screen_x}, {screen_y}) failed. Trying next...",
                )

        print(f"[ERROR] All {len(raw_candidates)} OpenCV candidates failed.")
        return False


# -------------------
# Public run functions
# -------------------
def run_llm() -> None:
    """Execute the Notepad automation task using the LLM launch function."""
    NotepadTask().automation_loop(NotepadTask().launch_llm)


def run_opencv() -> None:
    """Execute the Notepad automation task using the OpenCV launch function."""
    NotepadTask().automation_loop(NotepadTask().launch_opencv)
