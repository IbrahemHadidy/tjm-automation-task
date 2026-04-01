"""Provide utility functions for Windows DPI awareness and image preprocessing.

Manage system-level display settings, execution timing contexts, and
OpenCV-based image manipulation for the grounding engine.
"""

import logging
from collections import Counter
from typing import TYPE_CHECKING

import cv2
import numpy as np

from cv_strategy.constants import (
    BGR_TO_GRAY,
    BGR_TO_LAB,
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    COLOR_FUSED,
    COLOR_OCR,
    COLOR_VISUAL,
    DEFAULT_ICON_SIZE,
    INNER_THICKNESS,
    LAB_TO_BGR,
    OUTER_THICKNESS,
    TASKBAR_HEIGHT_PX,
    VIZ_MARKER_SIZE,
    VIZ_TEXT_OFFSET,
    VIZ_TEXT_SCALE,
    VIZ_TEXT_THICKNESS,
)

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from cv_strategy.models import Candidate, GroundingConfig

logger = logging.getLogger(__name__)


class ImageUtils:
    """Provide static utilities for loading, cropping, and enhancing screenshots."""

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
            lab = cv2.cvtColor(img, BGR_TO_LAB)
            channels = list(cv2.split(lab))
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            channels[0] = clahe.apply(channels[0])
            limg = cv2.merge(channels)
            return cv2.cvtColor(limg, LAB_TO_BGR)
        return img

    @staticmethod
    def detect_icon_sizes(
        roi: MatLike,
        config: GroundingConfig,
        top_n: int = 2,
    ) -> list[int]:
        """Detect the N most likely UI element scales using structural contour analysis.

        This method identifies common icon sizes by:
            1. Applying Canny edge detection.
            2. Performing Morphological Closing to 'seal' fragmented icon sub-components
               (e.g., merging a gear icon's teeth into a single 48px container).
            3. Filtering for square-ish contours to ignore text lines and window borders.
            4. Binning widths to account for anti-aliasing and rendering artifacts.

        Args:
            roi: The image region (OpenCV BGR) to analyze.
            config: Grounding configuration containing min/max icon width bounds.
            top_n: The maximum number of candidate size clusters to return.

        Returns:
            A list of detected icon widths in pixels, sorted by frequency.
            Returns [DEFAULT_ICON_SIZE] if no valid candidates are found.

        """
        min_w = config.min_icon_width
        max_w = config.max_icon_width

        # 1. Pre-process to "Solidify" UI Elements
        gray = cv2.cvtColor(roi, BGR_TO_GRAY)
        edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

        # SMARTS: Close gaps between edges to find the 'container'
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed,  # Use the 'closed' edges
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # 2. Filter for Square-ish containers
        valid_widths = []
        for c in contours:
            _x, _y, w, h = cv2.boundingRect(c)
            # Only accept widths in range that are roughly square (Aspect Ratio ~1.0)
            if min_w < w < max_w:
                aspect_ratio = w / float(h) if h > 0 else 0
                if 0.75 < aspect_ratio < 1.25:
                    # Binning: Round to nearest 4 to group 31px/32px/33px together
                    valid_widths.append((w // 4) * 4)

        if not valid_widths:
            return [DEFAULT_ICON_SIZE]

        # 3. Frequency Analysis on cleaned data
        common_peaks = Counter(valid_widths).most_common(top_n)

        # Return unique sizes, sorted by how common they are
        return [size for size, _ in common_peaks]

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
                color = COLOR_FUSED
            elif "OCR" in method_str:
                color = COLOR_OCR
            else:
                color = COLOR_VISUAL

            # 2. Draw Secondary Text Bounding Box (if fused/OCR)
            # This shows the specific area where the text was detected.
            t_bbox = c.extra.get("text_bbox")
            if t_bbox:
                tx, ty, tw, th = t_bbox
                # Draw as a thinner, dotted-style (multiple thin lines) box
                cv2.rectangle(
                    debug_view,
                    (tx, ty),
                    (tx + tw, ty + th),
                    (0, 0, 0),
                    OUTER_THICKNESS,
                )
                cv2.rectangle(
                    debug_view,
                    (tx, ty),
                    (tx + tw, ty + th),
                    color,
                    INNER_THICKNESS,
                )

            # 3. Primary Visual: Main Bounding Box
            if hasattr(c, "bbox") and c.bbox:
                bx, by, bw, bh = c.bbox
                cv2.rectangle(
                    debug_view,
                    (bx, by),
                    (bx + bw, by + bh),
                    (0, 0, 0),
                    OUTER_THICKNESS,
                )
                cv2.rectangle(
                    debug_view,
                    (bx, by),
                    (bx + bw, by + bh),
                    color,
                    INNER_THICKNESS,
                )
            elif not t_bbox:
                cv2.circle(
                    debug_view,
                    (c.x, c.y),
                    max(2, VIZ_MARKER_SIZE // 4),
                    color,
                    -1,
                )

            # 4. Label Preparation
            score_val = f"{c.score:.2f}"
            detected_text = c.extra.get("matched_text") or c.extra.get("text", "")

            main_label = f"#{i + 1} {method_str} [{score_val}]"
            sub_label = f"TEXT: '{detected_text}'" if detected_text else ""

            audit_label = ""
            if "FUSED" in method_str:
                # Extract the specific visual method used (e.g., tpl_edge, tpl_color)
                v_source = c.extra.get("visual_source")
                if v_source is not None:
                    v_name = getattr(v_source, "value", str(v_source))
                else:
                    v_name = "V"

                # Format: tpl_edge(0.85) + T(0.92)
                audit_label = f"{v_name}({c.img_score:.2f}) + T({c.txt_score:.2f})"

            # 5. Vertical Positioning
            is_at_top = c.y < 120
            y_offsets = (
                [VIZ_TEXT_OFFSET[1], VIZ_TEXT_OFFSET[1] * 2, VIZ_TEXT_OFFSET[1] * 3]
                if is_at_top
                else [
                    -VIZ_TEXT_OFFSET[1] * 3,
                    -VIZ_TEXT_OFFSET[1] * 2,
                    -VIZ_TEXT_OFFSET[1],
                ]
            )

            label_stack = [
                (main_label, y_offsets[0], VIZ_TEXT_SCALE[0]),
                (sub_label, y_offsets[1], VIZ_TEXT_SCALE[1]),
                (audit_label, y_offsets[2], VIZ_TEXT_SCALE[2]),
            ]

            # 6. Render Label Stack with Edge-Aware Clamping
            for text, y_off, scale in label_stack:
                if not text:
                    continue
                (tw, th), _ = cv2.getTextSize(text, font, scale, VIZ_TEXT_THICKNESS)
                t_x = c.x + VIZ_TEXT_OFFSET[0]
                if t_x + tw + 10 > canvas_w:
                    t_x = c.x - tw - VIZ_TEXT_OFFSET[0]
                t_x = max(5, min(t_x, canvas_w - tw - 5))
                t_y = max(th + 5, min(c.y + y_off, canvas_h - 10))
                t_pos = (t_x, t_y)

                rect_x1, rect_y1 = t_pos[0] - 5, t_pos[1] - th - 5
                rect_x2, rect_y2 = t_pos[0] + tw + 5, t_pos[1] + 5

                # Draw background rectangle for label
                cv2.rectangle(
                    debug_view,
                    (rect_x1, rect_y1),
                    (rect_x2, rect_y2),
                    (0, 0, 0),
                    -1,
                )
                # Draw colored border
                cv2.rectangle(
                    debug_view,
                    (rect_x1, rect_y1),
                    (rect_x2, rect_y2),
                    color,
                    INNER_THICKNESS,
                )
                # Render text
                cv2.putText(
                    debug_view,
                    text,
                    t_pos,
                    font,
                    scale,
                    (255, 255, 255),
                    VIZ_TEXT_THICKNESS,
                    cv2.LINE_AA,
                )

        return debug_view
