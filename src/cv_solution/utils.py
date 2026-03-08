"""Provide utility functions for Windows DPI awareness and image preprocessing.

Manage system-level display settings, execution timing contexts, and
OpenCV-based image manipulation for the grounding engine.
"""

import ctypes
import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import cv2

from cv_solution.constants import (
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    DEFAULT_ICON_SIZE,
    TASKBAR_HEIGHT_PX,
)
from cv_solution.models import PerfStat

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from cv2.typing import MatLike

    from cv_solution.models import Candidate, GroundingConfig

logger = logging.getLogger(__name__)


def set_high_dpi_awareness() -> None:
    """Configure the process to be DPI-aware on Windows.

    Attempt to set the highest level of DPI awareness to ensure coordinates
    captured via screenshots match the physical pixel grid.
    """
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception as e:
                logger.warning("Failed to set DPI awareness: %s", e)


@contextmanager
def time_block(name: str, stats_list: list[PerfStat]) -> Iterator[None]:
    """Measure the execution time of a code block and record performance stats.

    Args:
        name: Identifier for the operation being timed.
        stats_list: Collection where the resulting PerfStat will be appended.

    """
    t0 = time.time()
    yield
    duration = (time.time() - t0) * 1000
    stats_list.append(PerfStat(name, duration, 0))


class ImageUtils:
    """Provide static utilities for loading, cropping, and enhancing screenshots."""

    @staticmethod
    def load_screenshot(path: Path) -> MatLike:
        """Load an image from disk and validate its existence.

        Args:
            path: The filesystem path to the image file.

        Returns:
            The loaded image as an OpenCV MatLike object.

        Raises:
            FileNotFoundError: If the image cannot be loaded from the path.

        """
        img = cv2.imread(str(path))
        if img is None:
            msg = f"Failed to load screenshot at {path}"
            raise FileNotFoundError(msg)
        return img

    @staticmethod
    def crop_to_desktop(img: MatLike) -> MatLike:
        """Remove the Windows taskbar area based on the configured height.

        Args:
            img: The full-screen image to crop.

        Returns:
            A cropped image excluding the bottom taskbar region.

        """
        h, w = img.shape[:2]
        return img[0 : h - TASKBAR_HEIGHT_PX, 0:w]

    @staticmethod
    def enhance_contrast(img: MatLike) -> MatLike:
        """Apply CLAHE enhancement to improve recognition in low-contrast scenes.

        Adjust grayscale images directly or process the L-channel of BGR images
        using the CIELAB color space to preserve color while normalizing lighting.

        Args:
            img: The source image to enhance.

        Returns:
            The contrast-enhanced image.

        """
        if len(img.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            channels = list(cv2.split(lab))
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            channels[0] = clahe.apply(channels[0])
            limg = cv2.merge(channels)
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return img

    @staticmethod
    def detect_icon_size(roi: MatLike, config: GroundingConfig) -> int:
        """Detect the dominant desktop icon width using contour analysis.

        Analyze edge density and filter for rectangular contours within the
        expected icon width bounds defined in the configuration.

        Args:
            roi: The region of interest to analyze.
            config: Grounding configuration containing width bounds.

        Returns:
            The detected dominant icon width in pixels.

        """
        min_w = config.min_icon_width
        max_w = config.max_icon_width

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # PERF401: Optimized width extraction
        candidate_widths = [
            w for c in contours if min_w < (w := cv2.boundingRect(c)[2]) < max_w
        ]

        if not candidate_widths:
            return DEFAULT_ICON_SIZE

        # Use max with a count key for efficient dominance detection
        return max(set(candidate_widths), key=candidate_widths.count)

    @staticmethod
    def draw_candidates(
        canvas: cv2.typing.MatLike,
        candidates: list[Candidate],
    ) -> cv2.typing.MatLike:
        """Render high-contrast bounding boxes and smart labels for detections.

        This version draws both the primary visual bounding box and the
        secondary OCR/text bounding box (if available) to show fusion alignment.

        Args:
            canvas: The source image/frame to draw upon.
            candidates: A list of Candidate objects containing coordinates and metadata.

        Returns:
            A copy of the input canvas with clean detection overlays.

        """
        debug_view = canvas.copy()
        canvas_h, canvas_w = debug_view.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, c in enumerate(candidates):
            method_str = str(c.method.value).upper()

            # 1. Styling Logic
            if "FUSED" in method_str:
                color = (0, 255, 136)  # Success Green
            elif "OCR" in method_str:
                color = (0, 215, 255)  # Gold
            else:
                color = (255, 229, 0)  # Cyan

            # 2. Draw Secondary Text Bounding Box (if fused/OCR)
            # This shows the specific area where the text was detected.
            t_bbox = c.extra.get("text_bbox")
            if t_bbox:
                tx, ty, tw, th = t_bbox
                # Draw as a thinner, dotted-style (multiple thin lines) box
                cv2.rectangle(debug_view, (tx, ty), (tx + tw, ty + th), (0, 0, 0), 2)
                cv2.rectangle(debug_view, (tx, ty), (tx + tw, ty + th), color, 1)

            # 3. Primary Visual: Main Bounding Box
            if hasattr(c, "bbox") and c.bbox:
                bx, by, bw, bh = c.bbox
                # 4px Black Outer / 2px Color Inner
                cv2.rectangle(debug_view, (bx, by), (bx + bw, by + bh), (0, 0, 0), 4)
                cv2.rectangle(debug_view, (bx, by), (bx + bw, by + bh), color, 2)
            elif not t_bbox:
                # Fallback only if NO boxes exist
                cv2.circle(debug_view, (c.x, c.y), 5, color, -1)

            # 4. Label Preparation
            score_val = f"{c.score:.2f}"
            detected_text = c.extra.get("matched_text") or c.extra.get("text", "")

            main_label = f"#{i + 1} {method_str} [{score_val}]"
            sub_label = f"TEXT: '{detected_text}'" if detected_text else ""
            audit_label = (
                f"V({c.img_score:.2f}) + T({c.txt_score:.2f})"
                if "FUSED" in method_str
                else ""
            )

            # 5. Vertical Positioning
            is_at_top = c.y < 120
            y_offsets = [35, 55, 75] if is_at_top else [-65, -45, -25]

            label_stack = [
                (main_label, y_offsets[0], 0.55),
                (sub_label, y_offsets[1], 0.45),
                (audit_label, y_offsets[2], 0.40),
            ]

            # 6. Render Label Stack with Edge-Aware Clamping
            for text, y_off, scale in label_stack:
                if not text:
                    continue

                (tw, th), _ = cv2.getTextSize(text, font, scale, 1)

                t_x = c.x + 15
                if t_x + tw + 10 > canvas_w:
                    t_x = c.x - tw - 15

                t_x = max(5, min(t_x, canvas_w - tw - 5))
                t_y = max(th + 5, min(c.y + y_off, canvas_h - 10))
                t_pos = (t_x, t_y)

                rect_x1, rect_y1 = t_pos[0] - 5, t_pos[1] - th - 5
                rect_x2, rect_y2 = t_pos[0] + tw + 5, t_pos[1] + 5

                cv2.rectangle(
                    debug_view,
                    (rect_x1, rect_y1),
                    (rect_x2, rect_y2),
                    (0, 0, 0),
                    -1,
                )
                cv2.rectangle(
                    debug_view,
                    (rect_x1, rect_y1),
                    (rect_x2, rect_y2),
                    color,
                    1,
                )
                cv2.putText(
                    debug_view,
                    text,
                    t_pos,
                    font,
                    scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return debug_view
