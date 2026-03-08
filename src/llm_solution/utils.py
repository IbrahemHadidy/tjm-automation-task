"""Utility functions for AI vision processing and coordinate scaling."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from llm_solution.models import UIElementNode


class AiImageUtils:
    """Handles DPI awareness, coordinate transformations, and debug visualization.

    This class provides static helper methods to bridge the gap between
    normalized AI detections and physical screen pixels.
    """

    @staticmethod
    def setup_dpi_awareness() -> None:
        """Configure Windows to report actual pixels rather than scaled ones.

        This is critical for grounding tasks. Without this, on a 150% scaled
        display, a click at (1000, 1000) might actually land at (1500, 1500).
        """
        try:
            # Per-monitor awareness (highest level)
            ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
        except Exception:
            try:
                # Fallback for older Windows 8.1/10 versions
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                # Basic legacy awareness
                ctypes.windll.user32.SetProcessDPIAware()

    @staticmethod
    def scale_and_center(
        bbox: list[int],
        img_w: int,
        img_h: int,
        min_bbox_size: int = 5,
    ) -> tuple[list[int], list[int]]:
        """Convert normalized [0-1000] AI coordinates to absolute pixel values.

        Args:
            bbox: A list of [x1, y1, x2, y2] from the AI model.
            img_w: The actual width of the screenshot in pixels.
            img_h: The actual height of the screenshot in pixels.
            min_bbox_size: Minimum pixel dimension for the resulting box.

        Returns:
            A tuple containing:
                - center_coords: A list of [cx, cy] pixel coordinates.
                - dimensions: A list of [width, height] in pixels.

        """
        x1, y1, x2, y2 = bbox

        # Scale normalized float-like values to image dimensions
        px1 = int((x1 / 1000) * img_w)
        py1 = int((y1 / 1000) * img_h)
        px2 = int((x2 / 1000) * img_w)
        py2 = int((y2 / 1000) * img_h)

        # Enforce minimum size constraints to ensure clickable area
        px2 = max(px2, px1 + min_bbox_size)
        py2 = max(py2, py1 + min_bbox_size)

        cx = (px1 + px2) // 2
        cy = (py1 + py2) // 2

        return [cx, cy], [px2 - px1, py2 - py1]

    @staticmethod
    def draw_debug_results(
        image: Image.Image,
        nodes: list[UIElementNode],
    ) -> Image.Image:
        """Generate a high-contrast diagnostic overlay on the provided image.

        This method draws "hairline" boxes (magenta with black outlines) to
        ensure visibility against both light and dark UI themes.

        Args:
            image: The source PIL Image to draw upon.
            nodes: A list of processed UIElementNode detections.

        Returns:
            A new PIL Image containing the composite debug information.

        """
        # Convert to RGBA for professional layering
        canvas = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Handle font loading with fallback
        try:
            font = ImageFont.truetype("arialbd.ttf", 18)
        except OSError:
            font = ImageFont.load_default()

        for node in nodes:
            px, py = node["coords"]
            # Fallback for size if not provided by the detection
            w, h = node.get("size") or (40, 40)
            rank = node.get("rank", 0)
            score = node.get("score", 0.0)

            # Calculate box corners
            x1, y1 = px - w // 2, py - h // 2
            x2, y2 = px + w // 2, py + h // 2
            box = [x1, y1, x2, y2]

            # 1. High-Contrast Border: 4px Black under 2px Magenta
            draw.rectangle(box, outline=(0, 0, 0, 255), width=4)
            draw.rectangle(box, outline=(255, 0, 255, 255), width=2)

            # 2. Target Crosshair
            draw.line([px - 5, py, px + 5, py], fill=(255, 0, 255, 255), width=2)
            draw.line([px, py - 5, px, py + 5], fill=(255, 0, 255, 255), width=2)

            # 3. Smart Label Positioning
            label_text = f"#{rank} ({int(score * 100)}%)"
            left, top, right, bottom = draw.textbbox((0, 0), label_text, font=font)
            tw, th = right - left, bottom - top

            # Flip label to bottom if too close to top edge
            label_y = y2 + 5 if y1 < 30 else y1 - (th + 10)
            label_rect = [x1, label_y, x1 + tw + 10, label_y + th + 8]

            # Label Background
            draw.rectangle(label_rect, fill=(0, 0, 0, 255))
            draw.rectangle(
                [c + (1 if i < 2 else -1) for i, c in enumerate(label_rect)],
                fill=(255, 0, 255, 255),
            )

            draw.text((x1 + 5, label_y + 2), label_text, fill="white", font=font)

        # Merge and return as standard RGB
        return Image.alpha_composite(canvas, overlay).convert("RGB")
