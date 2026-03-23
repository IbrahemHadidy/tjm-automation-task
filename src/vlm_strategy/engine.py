"""Orchestrator for AI-based visual grounding using multi-modal vision.

This module provides the main entry point for resolving natural language
instructions into screen coordinates by managing the screenshot-to-AI pipeline.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from PIL import Image, ImageDraw

from screenshot_service import ScreenshotService
from vlm_strategy.client import AiClient
from vlm_strategy.models import UIElementNode
from vlm_strategy.prompts import DETECTION_PROMPT_TEMPLATE, VERIFICATION_PROMPT_TEMPLATE
from vlm_strategy.utils import AiImageUtils

if TYPE_CHECKING:
    from collections.abc import Callable

load_dotenv()


class AiGroundingEngine:
    """Coordinates visual grounding by connecting Client, Utils, and Screenshot services.

    This engine manages the retry logic, image preprocessing (like drawing red
    boundary boxes for window isolation), and post-action verification.
    """

    def __init__(
        self,
        model_id: str = "gemini-3.1-pro-preview",
    ) -> None:
        """Initialize components and performs system-level setup.

        Args:
            model_id: Gemini model identifier to use for vision tasks.

        Raises:
            ValueError: If GEMINI_API_KEY is not set in the environment.

        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            msg = "GEMINI_API_KEY not found in environment variables."
            raise ValueError(msg)

        self.client = AiClient(api_key=api_key, model_id=model_id)
        self.ss = ScreenshotService()

        # Default logger is print; can be overridden in resolve_coordinates
        self.log: Callable[[str], None] = print

        self.last_raw_image: Image.Image | None = None
        self.last_debug_image: Image.Image | None = None

    def resolve_coordinates(
        self,
        instruction: str,
        target_window: str = "Desktop",
        reference_image_path: str | None = None,
        logger_callback: Callable[[str], None] | None = None,
        max_retries: int = 3,
        *,
        verify_after_action: bool = False,
        restore_workspace: bool = False,
    ) -> list[UIElementNode]:
        """Identify UI elements on screen based on a natural language instruction.

        Args:
            instruction: The text description of the element to find.
            target_window: Title of the window to search within.
            reference_image_path: Path to a visual anchor/example image.
            logger_callback: Optional function to handle status logs.
            max_retries: Number of AI attempts if parsing fails.
            verify_after_action: Whether to perform a secondary crop check.
            restore_workspace: Whether to snapshot and restore workspace before/after capture.

        Returns:
            A list of UIElementNode objects found, sorted by model rank.

        """
        if logger_callback:
            self.log = logger_callback

        # 1. Capture Scope (PIL Image + Window Context)
        scope_prompt, exclusion_context, window_rect, raw_img = self._capture_scope(
            target_window,
            restore_workspace=restore_workspace,
        )

        # 2. Prepare AI input image (Draws boundary rectangles if needed)
        ai_input_image = self._prepare_ai_image(raw_img, window_rect)

        # 3. Build AI prompt using centralized templates
        ref_context = ""
        if reference_image_path and Path(reference_image_path).exists():
            ref_context = (
                f"- Only return matches that visually match the reference: "
                f"{reference_image_path}"
            )

        prompt = DETECTION_PROMPT_TEMPLATE.format(
            instruction=instruction,
            scope_prompt=scope_prompt,
            exclusion_context=exclusion_context,
            ref_context=ref_context,
        )

        # 4. Execute detection pipeline
        return self._detect_elements_from_image(
            ai_input_image,
            prompt,
            max_retries=max_retries,
            verify_after_action=verify_after_action,
            instruction=instruction,
        )

    def _capture_scope(
        self,
        target_window: str,
        *,
        restore_workspace: bool = False,
    ) -> tuple[str, str, tuple[int, int, int, int] | None, Image.Image]:
        """Capture the current screen state and define search boundaries.

        This method optionally snapshots and restores the workspace to ensure
        a clean visual environment before/after capture.

        Args:
            target_window: The target window name or 'Desktop'.
            restore_workspace: Whether to snapshot and restore workspace
            around the capture for a clean visual state.

        Returns:
            A tuple of (scope_text, exclusion_text, window_rect, image):
                scope_text: Human-readable description of the captured area.
                exclusion_text: Text describing areas to ignore (e.g., taskbar).
                window_rect: Coordinates of the target window (x, y, w, h) or None.
                image: The captured PIL.Image of the screen or window.

        """
        if restore_workspace:
            self.ss.snapshot_workspace()

        response: tuple[str, str, tuple[int, int, int, int] | None, Image.Image]

        if target_window == "Desktop":
            self.log("[INFO] Capturing Full Desktop...")
            img = self.ss.capture_desktop()
            response = (
                "the entire desktop",
                "- Ignore the Windows Taskbar and system tray icons.",
                None,
                img,
            )
        else:
            self.log(f"[INFO] Isolating {target_window}...")
            img, window_rect = self.ss.capture_app_window(target_window)
            response = "the area inside the RED rectangle", "", window_rect, img

        if restore_workspace:
            self.ss.restore_workspace()

        return response

    def _prepare_ai_image(
        self,
        image: Image.Image,
        window_rect: tuple[int, int, int, int] | None,
    ) -> Image.Image:
        """Draws visual cues (like red rectangles) to guide the AI's attention.

        Args:
            image: The raw PIL screenshot.
            window_rect: The coordinates [x, y, w, h] of the target window.

        Returns:
            A PIL image modified with visual grounding cues.

        """
        canvas = image.copy().convert("RGB")
        if window_rect:
            draw = ImageDraw.Draw(canvas)
            l, t, w, h = window_rect
            draw.rectangle([l, t, l + w, t + h], outline="red", width=10)
        return canvas

    def _detect_elements_from_image(
        self,
        ai_vision_img: Image.Image,
        prompt: str,
        max_retries: int,
        instruction: str,
        *,
        verify_after_action: bool,
    ) -> list[UIElementNode]:
        """Orchestrates the AI inference loop and result scaling.

        Args:
            ai_vision_img: The processed image sent to the AI.
            prompt: The formatted instruction string.
            max_retries: Maximum number of API attempts.
            instruction: Original user instruction for verification.
            verify_after_action: Flag to trigger secondary validation.

        Returns:
            A list of localized UIElementNode results.

        """
        img_w, img_h = ai_vision_img.size
        self.last_raw_image = ai_vision_img.copy()

        for attempt in range(max_retries):
            self.log(f"[INFO] AI Attempt {attempt + 1}/{max_retries}...")
            try:
                results = self.client.generate_detection(prompt, ai_vision_img)
                if not results:
                    continue

                nodes: list[UIElementNode] = []
                for r in results:
                    coords, size = AiImageUtils.scale_and_center(
                        bbox=r["bbox"],
                        img_w=img_w,
                        img_h=img_h,
                    )

                    node = UIElementNode(
                        coords=coords,
                        score=float(r.get("score", 0.0)),
                        area=r.get("area", ""),
                        neighbors=r.get("neighbors", []),
                        rank=int(r.get("rank", 1)),
                        size=size,
                    )
                    nodes.append(node)

                    if verify_after_action:
                        self._verify_detection(coords, instruction, img_w, img_h)

                # Generate debug visualization for the laboratory
                self.last_debug_image = AiImageUtils.draw_debug_results(
                    ai_vision_img,
                    nodes,
                )

                self.log(f"[SUCCESS] AI found {len(nodes)} candidates.")

            except Exception as e:
                self.log(f"[ERROR] AI Engine Pipeline failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

            return nodes

        return []

    def _verify_detection(
        self,
        coords: list[int],
        instruction: str,
        img_w: int,
        img_h: int,
    ) -> str:
        """Perform a secondary 'eyes-on' check on a localized crop.

        This helps eliminate false positives by asking the AI to focus exclusively
        on the area it just identified.

        Args:
            coords: [x, y] center of the detected element.
            instruction: The original search query.
            img_w: Full image width.
            img_h: Full image height.

        Returns:
            The verification string from the model (e.g., 'is_target').

        """
        self.log("[INFO] Performing post-action verification...")

        x, y = coords
        margin = 100
        box = (
            max(0, x - margin),
            max(0, y - margin),
            min(img_w, x + margin),
            min(img_h, y + margin),
        )

        current_img = self.ss.capture_desktop()
        verify_crop = current_img.crop(box)
        verify_prompt = VERIFICATION_PROMPT_TEMPLATE.format(instruction=instruction)

        resp = self.client.client.models.generate_content(
            model=self.client.model_id,
            contents=[verify_prompt, verify_crop],
        )

        result = resp.text or "unknown"
        self.log(f"[INFO] Verification result: {result}")
        return result
