"""Engine for AI-based visual grounding and action verification."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from dotenv import load_dotenv
from google import genai
from PIL import Image, ImageDraw

from screenshot_service import ScreenshotService

if TYPE_CHECKING:
    from collections.abc import Callable

load_dotenv()


class UIElementNode(TypedDict):
    """Represent a localized UI element identified by the AI."""

    coords: list[int]  # [x, y]
    score: float


class AiGroundingEngine:
    """Coordinates visual grounding using multi-modal AI vision."""

    def __init__(self, model_id: str = "gemini-3-flash-preview") -> None:
        """Initialize the AI client and screenshot utility."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            msg = "GEMINI_API_KEY not found in environment variables."
            raise ValueError(msg)

        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.ss = ScreenshotService()
        self.log: Callable[[str], None] = print
        self.cleanup()

    def resolve_coordinates(  # noqa: PLR0913
        self,
        instruction: str,
        target_window: str = "Entire Desktop",
        reference_image_path: str | None = None,
        logger_callback: Callable[[str], None] | None = None,
        max_retries: int = 3,
        scale_to_pixels: bool = True,  # noqa: FBT001, FBT002
    ) -> list[UIElementNode]:
        """Identify UI elements and convert normalized AI coordinates to pixel values."""
        if logger_callback:
            self.log = logger_callback

        temp_ss = Path("capture_state.png")
        ai_input_path = Path("ai_vision_input.png")
        window_rect = None

        # Determine capture scope
        if target_window == "Entire Desktop":
            self.log("[INFO] Capturing Full Desktop...")
            self.ss.capture_desktop(temp_ss)
            scope_prompt = "the entire desktop"
            exclusion_context = "- Ignore the Windows Taskbar and system tray icons."
        else:
            self.log(f"[INFO] Isolating {target_window} against black background...")
            _, window_rect = self.ss.capture_app_window(target_window, temp_ss)
            scope_prompt = "the area inside the RED rectangle"
            exclusion_context = ""

        # Process image for AI input
        shutil.copy(temp_ss, "raw_capture.png")
        with Image.open(temp_ss) as img:
            if window_rect:
                ai_img = img.copy()
                draw = ImageDraw.Draw(ai_img)
                l, t, w, h = window_rect  # noqa: E741
                draw.rectangle([l, t, l + w, t + h], outline="red", width=10)
                ai_img.save(ai_input_path)
            else:
                shutil.copy(temp_ss, ai_input_path)

        # Build context-aware prompt
        ref_context = ""
        if reference_image_path and Path(reference_image_path).exists():
            ref_context = "- Only return matches that visually match the provided reference image."

        prompt = (
            f"Task: Identify ALL instances of: '{instruction}'\n"
            f"Scope: Search ONLY within {scope_prompt}.\n"
            f"{exclusion_context}\n"
            f"{ref_context}\n\n"
            "Requirements:\n"
            "1. COORDINATES: "
            "Return [x, y] normalized (0-1000) relative to the FULL IMAGE.\n"
            "2. SCORING LOGIC:\n"
            " - Score 1.0: Exact match (e.g., The 'View' menu in the header).\n"
            " - Score 0.7-0.9: Close variation. Examples:\n"
            "    * Text variation: Found 'Notepad++' or 'Notepad2' when 'Notepad' was requested.\n"
            "    * Menu variation: Found 'Zoom' or 'Status Bar' when 'View' was requested (related sub-items).\n"
            " - Score < 0.5: Significant visual differences / Irrelevant. Examples:\n"
            "    * Icon/Text Mismatch: The requested text has nothing to do with the icon found (e.g., Finding a 'Trash' icon when looking for 'Format').\n"
            "    * Role mismatch: Finding a minimize button when looking for the 'Edit' text menu.\n"
            "    * State mismatch: Finding the 'Undo' button, but it is greyed out/disabled.\n"
            "    * Content mismatch: Finding the word 'Notepad' written inside the document body instead of the application title.\n"
            "3. JSON ONLY: "
            'JS[{"coords": [x, y], "score": float}]\n'
        )

        # Execute AI request with retry logic
        with Image.open(ai_input_path) as ai_vision_img:
            img_w, img_h = ai_vision_img.size

            for attempt in range(max_retries):
                self.log(
                    f"[INFO] Attempt {attempt + 1}/{max_retries} to {self.model_id}...",
                )
                try:
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=[prompt, ai_vision_img],
                    )

                    results = self._parse_json_list(response.text or "")
                    if results:
                        # Transform coordinates if requested
                        if scale_to_pixels:
                            for node in results:
                                nx, ny = node["coords"]
                                node["coords"] = [
                                    int((nx / 1000) * img_w),
                                    int((ny / 1000) * img_h),
                                ]

                        self.log(f"[SUCCESS] Received {len(results)} nodes.")
                        return results

                    self.log(f"[RETRY] Attempt {attempt + 1} yielded no results.")
                    time.sleep(1)
                except Exception as e:
                    self.log(f"[ERROR] API Call failed: {e}")

        return []

    def _parse_json_list(self, text: str) -> list[UIElementNode]:
        """Extract and parse JSON content from the AI response string."""
        try:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            return json.loads(match.group()) if match else []
        except (json.JSONDecodeError, AttributeError):
            return []

    def cleanup(self) -> None:
        """Delete temporary image artifacts from the workspace."""
        temp_files = [
            "capture_state.png",
            "ai_vision_input.png",
            "raw_capture.png",
            "verify_state.png",
        ]
        for file in temp_files:
            path = Path(file)
            if path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    self.log(f"[ERROR] Could not delete {file}: {e}")
