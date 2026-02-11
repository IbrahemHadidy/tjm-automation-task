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

LLM_INSTRUCTION = "Notepad shortcut"
OPENCV_TEXT_QUERY = "Notepad"
PROJECT_DIR = Path.home() / "Desktop" / "tjm-project"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

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
        wins = [
            w
            for w in gw.getWindowsWithTitle("Notepad")
            if w.visible and not w.isMinimized
        ]
        return wins[0] if wins else None

    # -------------------
    # Shared helpers (DRY)
    # -------------------
    def _reset_desktop(self) -> None:
        pyautogui.hotkey("win", "m")
        time.sleep(1.5)
        pyautogui.click(1, 1)

    def _verify_and_launch(
        self,
        *,
        x: int,
        y: int,
        score: float,
        source: str,
        min_score: float = 0.5,
    ) -> bool:
        if score < min_score:
            return False

        print(
            f"[INFO] Attempting launch with {source} candidate at ({x}, {y}) (Score: {score})",
        )
        pyautogui.doubleClick(x, y)

        for _ in range(6):
            if self._get_active_notepad():
                print(f"[SUCCESS] Notepad launched via {source} candidate.")
                return True
            time.sleep(0.5)

        print(f"[WARN] {source} candidate at ({x}, {y}) failed.")
        return False

    # -------------------
    # API helpers
    # -------------------
    def _fetch_posts(self) -> list[Post]:
        """Fetch post data from the API with 3 retries and exponential backoff."""
        for attempt in range(3):
            try:
                print(f"[INFO] Fetching posts (Attempt {attempt + 1}/3)...")
                resp = requests.get(self.posts_api, timeout=10)
                resp.raise_for_status()
                return self._validate_posts(resp.json())
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

        pyautogui.hotkey("ctrl", "s")  # Trigger Save

        # Wait for Save As dialog
        save_dialog = None
        for _ in range(10):
            dialogs = [w for w in gw.getWindowsWithTitle("Save As") if w.visible]
            if dialogs:
                save_dialog = dialogs[0]
                break
            time.sleep(0.5)

        if not save_dialog:
            print(f"[WARNING] Save dialog failed for post {post['id']}.")
            return

        file_path = str(PROJECT_DIR / f"post_{post['id']}.txt")
        pyautogui.hotkey("alt", "n")
        pyautogui.hotkey("ctrl", "a")
        pyautogui.press("backspace")
        pyperclip.copy(file_path)
        pyautogui.hotkey("ctrl", "v")
        pyautogui.press("enter")

        time.sleep(0.7)
        if any(w.visible for w in gw.getWindowsWithTitle("Confirm Save As")):
            pyautogui.press("y")

        pyautogui.hotkey("ctrl", "w")
        with contextlib.suppress(Exception):
            win.close()

        print(f"[COMPLETE] Processed post {post['id']} and cleaned up.")

    # -------------------
    # Launchers
    # -------------------
    def launch_llm(self) -> bool:
        """Use AI grounding engine to locate Notepad icon and double-click it."""
        self._reset_desktop()

        def engine_logger(msg: str) -> None:
            level = "INFO"
            if "[SUCCESS]" in msg:
                level = "SUCCESS"
            elif "[ERROR]" in msg:
                level = "ERROR"
            clean = (
                msg.replace("[INFO]", "")
                .replace("[SUCCESS]", "")
                .replace(
                    "[ERROR]",
                    "",
                )
            )
            print(f"[ENGINE][{level}] {clean.strip()}")

        try:
            results = self.llm_engine.resolve_coordinates(
                instruction=LLM_INSTRUCTION,
                target_window="Entire Desktop",
                scale_to_pixels=True,
                verify_after_action=False,
                logger_callback=engine_logger,
            )

            if not results:
                print("[ERROR] No AI matches returned.")
                return False

            results.sort(key=lambda n: (-n.get("score", 0.0), n.get("rank", 1)))

            for node in results:
                x, y = node.get("coords", [0, 0])
                if self._verify_and_launch(
                    x=x,
                    y=y,
                    score=node.get("score", 0.0),
                    source="AI",
                ):
                    return True

            print(f"[ERROR] All {len(results)} AI candidates failed.")
            return False

        finally:
            self.llm_engine.cleanup()

    def launch_opencv(self) -> bool:
        """Use OpenCV grounding engine to locate Notepad icon and double-click it."""
        self._reset_desktop()

        temp_ss = Path("grounding_temp.png")
        pyautogui.screenshot(str(temp_ss))

        try:
            raw = self.cv_engine.locate_elements(
                screenshot_path=temp_ss,
                icon_image=ICON_PATH,
                text_query=OPENCV_TEXT_QUERY,
                threshold=0.5,
                config={
                    "use_ocr": True,
                    "use_color": True,
                    "use_multiscale": True,
                    "num_cores": 8,
                },
            )

            raw.sort(key=lambda c: c.score, reverse=True)

            screen_w, screen_h = pyautogui.size()
            img = cv2.imread(str(temp_ss))
            if img is None:
                print("[ERROR] Could not read screenshot.")
                return False

            img_h, img_w = img.shape[:2]
            sx, sy = screen_w / img_w, screen_h / img_h

            for cand in raw:
                x = int(cand.x * sx)
                y = int(cand.y * sy)
                if self._verify_and_launch(
                    x=x,
                    y=y,
                    score=cand.score,
                    source="OpenCV",
                ):
                    return True

            print(f"[ERROR] All {len(raw)} OpenCV candidates failed.")
            return False

        finally:
            temp_ss.unlink(missing_ok=True)


# -------------------
# Public run functions
# -------------------
def run_llm() -> None:
    """Execute the Notepad automation task using the LLM launch function."""
    NotepadTask().automation_loop(NotepadTask().launch_llm)


def run_opencv() -> None:
    """Execute the Notepad automation task using the OpenCV launch function."""
    NotepadTask().automation_loop(NotepadTask().launch_opencv)
