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


class AIDetection(TypedDict):
    """Raw detection result returned by the vision model."""

    bbox: list[int]  # [x1, y1, x2, y2] normalized 0-1000
    score: float
    area: str
    neighbors: list[str]
    rank: int


class UIElementNode(TypedDict):
    """Represent a localized UI element identified by the AI."""

    coords: list[int]  # [x, y] center pixel coordinates
    score: float  # confidence
    area: str  # UI area description
    neighbors: list[str]  # nearby elements
    rank: int  # ranking (1 = most likely)
    size: list[int] | None  # [width, height] in pixels


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

        # Minimum bbox size in pixels to avoid zero-size errors
        self.min_bbox_size = 5

    def resolve_coordinates(
        self,
        instruction: str,
        target_window: str = "Entire Desktop",
        reference_image_path: str | None = None,
        logger_callback: Callable[[str], None] | None = None,
        max_retries: int = 3,
        *,
        verify_after_action: bool = False,
        scale_to_pixels: bool = True,
    ) -> list[UIElementNode]:
        """Identify UI elements and convert normalized AI coordinates to pixel values."""
        if logger_callback:
            self.log = logger_callback

        temp_ss = Path("capture_state.png")
        ai_input_path = Path("ai_vision_input.png")

        # Capture scope
        scope_prompt, exclusion_context, window_rect = self._capture_scope(
            target_window,
            temp_ss,
        )

        shutil.copy(temp_ss, "raw_capture.png")

        # Prepare AI input image
        self._prepare_ai_image(temp_ss, ai_input_path, window_rect)

        # Reference context
        ref_context = ""
        if reference_image_path and Path(reference_image_path).exists():
            ref_context = "- Only return matches that visually match the provided reference image."

        # Build AI prompt
        prompt = self._build_detection_prompt(
            instruction,
            scope_prompt,
            exclusion_context,
            ref_context,
        )

        # Detect elements from AI
        return self._detect_elements_from_image(
            ai_input_path,
            prompt,
            max_retries=max_retries,
            scale_to_pixels=scale_to_pixels,
            verify_after_action=verify_after_action,
            instruction=instruction,
            temp_ss=temp_ss,
        )

    # ----------------------- Scope & Image Prep -----------------------

    def _capture_scope(
        self,
        target_window: str,
        temp_ss: Path,
    ) -> tuple[str, str, tuple[int, int, int, int] | None]:
        """Capture the screen or window and return scope prompt, exclusion text, and window rect."""
        window_rect = None
        if target_window == "Entire Desktop":
            self.log("[INFO] Capturing Full Desktop...")
            self.ss.capture_desktop(temp_ss)
            return (
                "the entire desktop",
                "- Ignore the Windows Taskbar and system tray icons.",
                None,
            )
        self.log(f"[INFO] Isolating {target_window} against black background...")
        _, window_rect = self.ss.capture_app_window(target_window, temp_ss)
        return "the area inside the RED rectangle", "", window_rect

    def _prepare_ai_image(
        self,
        temp_ss: Path,
        ai_input_path: Path,
        window_rect: tuple[int, int, int, int] | None,
    ) -> None:
        """Draw red rectangle if window captured and save AI input image."""
        with Image.open(temp_ss) as img:
            if window_rect:
                ai_img = img.copy()
                draw = ImageDraw.Draw(ai_img)
                l, t, w, h = window_rect
                draw.rectangle([l, t, l + w, t + h], outline="red", width=10)
                ai_img.save(ai_input_path)
            else:
                shutil.copy(temp_ss, ai_input_path)

    # ----------------------- AI Prompt -----------------------

    def _build_detection_prompt(
        self,
        instruction: str,
        scope_prompt: str,
        exclusion_context: str,
        ref_context: str,
    ) -> str:
        """Return an AI prompt that enforces research-paper-style position inference with JSON output."""
        return (
            f"Position Inference Prompt:\n"
            f"You are tasked with identifying UI elements that match the following instruction: '{instruction}'\n"
            f"Scope: Search ONLY within {scope_prompt}.\n"
            f"{exclusion_context}\n"
            f"{ref_context}\n\n"
            "Important Guidelines:\n"
            "1. The target element is guaranteed to exist in the screenshot.\n"
            "2. Make references specific and unambiguous (no generic terms like 'window' or 'icons').\n"
            "3. List all likely candidates in descending order of probability.\n"
            "4. Do NOT propose operations that would change the screenshot.\n"
            "5. Always include context: the UI area and up to 3 neighboring elements.\n"
            "6. Provide a rank for each candidate (1 = most likely).\n\n"
            "Output format (JSON ONLY):\n"
            "[\n"
            "  {\n"
            '    "bbox": [x1, y1, x2, y2],  # normalized coordinates 0-1000\n'
            '    "score": 0.0-1.0,          # confidence score\n'
            '    "area": "<precise UI area description>",\n'
            '    "neighbors": ["neighbor1", "neighbor2", "neighbor3"],\n'
            '    "rank": 1\n'
            "  },\n"
            "  ...\n"
            "]\n\n"
            "Notes:\n"
            "- If uncertain, still return candidates with lower scores.\n"
            "- Do not return free-text; JSON array only.\n"
            "- Ensure bounding boxes tightly surround the clickable region.\n"
            "- Confidence scoring: 1.0 = exact match, 0.7-0.9 = close variation, <0.5 = irrelevant.\n"
        )

    # ----------------------- AI Detection -----------------------

    def _detect_elements_from_image(
        self,
        ai_input_path: Path,
        prompt: str,
        max_retries: int,
        instruction: str,
        temp_ss: Path,
        *,
        scale_to_pixels: bool,
        verify_after_action: bool,
    ) -> list[UIElementNode]:
        """Call AI model, parse results, scale coordinates, optionally verify detection."""
        with Image.open(ai_input_path) as ai_vision_img:
            img_w, img_h = ai_vision_img.size
            for attempt in range(max_retries):
                self.log(f"[INFO] Attempt {attempt + 1}/{max_retries}...")
                try:
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=[prompt, ai_vision_img],
                    )
                    results = self._parse_json_list(response.text or "")
                    if not results:
                        self.log(f"[RETRY] Attempt {attempt + 1} yielded no results.")
                        time.sleep(1)
                        continue

                    nodes: list[UIElementNode] = []
                    for r in results:
                        # Scale bbox to center pixel coordinates
                        coords, size = self._scale_and_center(
                            r["bbox"],
                            img_w,
                            img_h,
                            scale_to_pixels,
                        )

                        # Create the UIElementNode with extra context
                        node: UIElementNode = UIElementNode(
                            coords=coords,
                            score=float(r.get("score", 0.0)),
                            area=r.get("area", ""),
                            neighbors=r.get("neighbors", []),
                            rank=int(r.get("rank", 1)),
                            size=size,
                        )
                        nodes.append(node)

                        # Optional verification
                        if verify_after_action:
                            self._verify_detection(
                                coords,
                                instruction,
                                img_w,
                                img_h,
                                temp_ss,
                            )

                    self.log(f"[SUCCESS] Received {len(nodes)} nodes.")
                except Exception as e:
                    self.log(f"[ERROR] API Call failed: {e}")
                else:
                    return nodes

        return []

    def _validate_ai_detection(self, r: dict) -> bool:
        """Check that detection contains valid bbox, score, and optional fields."""
        try:
            bbox = r["bbox"]
            score = float(r["score"])
            if len(bbox) != 4:
                return False
            if not all(isinstance(v, (int, float)) for v in bbox):
                return False
            if not (0 <= score <= 1):
                return False
        except (KeyError, TypeError, ValueError):
            return False
        else:
            return True

    def _scale_and_center(
        self,
        bbox: list[int],
        img_w: int,
        img_h: int,
        scale_to_pixels: bool,  # noqa: FBT001
    ) -> tuple[list[int], list[int]]:
        """Convert normalized bbox to pixel coordinates and compute center."""
        x1, y1, x2, y2 = bbox
        if scale_to_pixels:
            x1 = int((x1 / 1000) * img_w)
            y1 = int((y1 / 1000) * img_h)
            x2 = int((x2 / 1000) * img_w)
            y2 = int((y2 / 1000) * img_h)

        x2 = max(x2, x1 + self.min_bbox_size)
        y2 = max(y2, y1 + self.min_bbox_size)
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = min(x2, img_w)
        y2 = min(y2, img_h)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        width = x2 - x1
        height = y2 - y1
        return [cx, cy], [width, height]

    # ----------------------- Verification -----------------------

    def _verify_detection(
        self,
        coords: list[int],
        instruction: str,
        img_w: int,
        img_h: int,
        temp_ss: Path,
    ) -> str:
        """Verify that the detected element is correctly located and return verification result."""
        self.log("[INFO] Performing optional post-action verification...")
        crop_box = self._get_verification_crop(coords, img_w, img_h)
        self.ss.capture_desktop(temp_ss)
        with Image.open(temp_ss) as verify_img:
            verify_crop = verify_img.crop(crop_box)
            verify_prompt = (
                "You are given a cropped screenshot. Evaluate if the marked element matches the target.\n"
                "Return JSON ONLY: {'result': 'is_target' | 'target_elsewhere' | 'target_not_found', 'new_instruction': '<optional>'}\n"
                f"Instruction: {instruction}"
            )
            resp = self.client.models.generate_content(
                model=self.model_id,
                contents=[verify_prompt, verify_crop],
            )

            resp_text = resp.text or "{}"
            try:
                result_obj = json.loads(resp_text)
                return result_obj.get("result", "target_not_found")
            except (json.JSONDecodeError, TypeError):
                return "target_not_found"

    def _get_verification_crop(
        self,
        coords: list[int],
        img_w: int,
        img_h: int,
        margin: int = 50,
    ) -> tuple[int, int, int, int]:
        """Return a generous crop around the coordinates."""
        x, y = coords
        left = max(0, x - margin)
        top = max(0, y - margin)
        right = min(img_w, x + margin)
        bottom = min(img_h, y + margin)
        return (left, top, right, bottom)

    # ----------------------- Utility -----------------------

    def _parse_json_list(self, text: str) -> list[dict]:
        """Extract and parse JSON array from AI response."""
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
