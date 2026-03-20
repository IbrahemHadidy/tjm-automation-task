"""Processor for global text search and local label recovery using Tesseract OCR.

Manage multi-modal OCR pipelines that handle both screen-wide text detection
and targeted label recovery around visual icon candidates.
"""

import logging
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

import cv2
import pytesseract
from cv_strategy.constants import (
    INTERPOLATION_UP,
    MAX_8BIT_VALUE,
    OCR_ENGINE_MODE,
    OCR_EXACT_MATCH_SCORE,
    OCR_FUZZY_MATCH_THRESHOLD,
    OCR_GLOBAL_SEARCH_MODES,
    OCR_GLOBAL_UPSCALE_FACTOR,
    OCR_LOCAL_UPSCALE_FACTOR,
    OCR_MIN_CONFIDENCE,
    OCR_MIN_TOKEN_LENGTH,
    OCR_MODE_CUBIC_UPSCALE,
    OCR_MODE_INVERTED_OTSU,
    OCR_MODE_OTSU,
    OCR_MODE_TOPHAT_UPSCALE_A,
    OCR_MODE_TOPHAT_UPSCALE_B,
    OCR_MORPH_KERNEL_SIZE,
    OCR_RECOVERY_PENALTY,
    OCR_RECOVERY_THRESHOLD,
    OCR_SIMILARITY_WEIGHT,
    OCR_SUBSTRING_BASE_SCORE,
    OCR_SUBSTRING_MATCH_WEIGHT,
    PSM_SINGLE_LINE,
    RECOVERY_HORIZONTAL_PAD_FACTOR,
    RECOVERY_VERTICAL_EXTEND_FACTOR,
    RECOVERY_VERTICAL_OFFSET_PX,
)
from cv_strategy.models import Candidate, DetectionMethod
from cv_strategy.utils import ImageUtils

if TYPE_CHECKING:
    from collections.abc import Callable

    from cv2.typing import MatLike
    from cv_strategy.models import GroundingConfig

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handle text-based grounding via Tesseract.

    Provide methods for global screen scraping and localized Region of Interest
    (ROI) text verification to confirm icon identities.
    """

    def __init__(self, tesseract_path: str, config: GroundingConfig) -> None:
        """Initialize the OCR engine with the provided Tesseract executable path.

        Args:
            tesseract_path: Filesystem path to the Tesseract binary.
            config: Configuration object for OCR behavior and thresholds.

        """
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.config = config

    def search_global(
        self,
        roi: MatLike,
        query: str,
        log_callback: Callable[..., Any],
    ) -> list[Candidate]:
        """Run multiple preprocessing modes to find text across the entire image.

        Args:
            roi: The image or region to search for text.
            query: The text string to locate.
            log_callback: Function to report progress and show debug images.

        Returns:
            A list of Candidates where the text was found.

        """
        results: list[Candidate] = []
        normalized_query = query.strip().lower()

        for mode_id in OCR_GLOBAL_SEARCH_MODES:
            # UI Feedback: Show the preprocessed state before OCR hits it
            processed_image = self._preprocess_for_mode(roi, mode_id)
            if self.config.use_adaptive:
                processed_image = ImageUtils.enhance_contrast(processed_image)

            log_callback(f"OCR: Scanning Mode {mode_id}...", processed_image)

            psm_mode = (
                PSM_SINGLE_LINE
                if mode_id == OCR_MODE_TOPHAT_UPSCALE_B
                else self.config.psm
            )

            try:
                ocr_data = pytesseract.image_to_data(
                    processed_image,
                    output_type=pytesseract.Output.DICT,
                    config=f"--oem {OCR_ENGINE_MODE} --psm {psm_mode}",
                    lang=self.config.ocr_lang,
                )

                upscale_modes = [
                    OCR_MODE_CUBIC_UPSCALE,
                    OCR_MODE_TOPHAT_UPSCALE_A,
                    OCR_MODE_TOPHAT_UPSCALE_B,
                ]
                scale_factor = (
                    OCR_GLOBAL_UPSCALE_FACTOR if mode_id in upscale_modes else 1.0
                )

                new_hits = self._parse_tesseract_data(
                    ocr_data,
                    normalized_query,
                    scale_factor,
                    DetectionMethod.OCR_GLOBAL,
                    extra={"ocr_mode": mode_id, "upscaled": scale_factor > 1.0},
                )
                results.extend(new_hits)

            except Exception:
                logger.exception("OCR mode %s failed", mode_id)

        return results

    def _parse_tesseract_data(
        self,
        data: dict,
        query: str,
        scale: float,
        method: DetectionMethod,
        extra: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        """Extract candidate coordinates from Tesseract's dictionary output.

        Args:
            data: Dictionary returned by pytesseract.image_to_data.
            query: The target text to match against.
            scale: The scaling factor applied to the image before OCR.
            method: The detection strategy used.
            extra: Optional metadata to attach to candidates.

        Returns:
            List of parsed Candidate objects.

        """
        candidates = []
        for i, text in enumerate(data.get("text", [])):
            found_text = str(text).strip().lower()
            confidence = float(data["conf"][i])

            if (
                confidence < OCR_MIN_CONFIDENCE
                or len(found_text) < OCR_MIN_TOKEN_LENGTH
            ):
                continue

            similarity = SequenceMatcher(None, query, found_text).ratio()

            if (
                found_text == query
                or query in found_text
                or similarity > OCR_FUZZY_MATCH_THRESHOLD
            ):
                if found_text == query:
                    score = OCR_EXACT_MATCH_SCORE
                else:
                    score = OCR_SUBSTRING_BASE_SCORE + (
                        OCR_SIMILARITY_WEIGHT * similarity
                    )

                center_x = int((data["left"][i] + data["width"][i] / 2) / scale)
                center_y = int((data["top"][i] + data["height"][i] / 2) / scale)

                # Store the actual OCR bounding box
                ocr_bbox = (
                    int(data["left"][i] / scale),
                    int(data["top"][i] / scale),
                    int(data["width"][i] / scale),
                    int(data["height"][i] / scale),
                )

                # Merge passed extra with match specific data
                meta = extra.copy() if extra else {}
                meta.update(
                    {"text": found_text, "conf": confidence, "text_bbox": ocr_bbox},
                )

                candidates.append(
                    Candidate(
                        x=center_x,
                        y=center_y,
                        score=score,
                        method=method,
                        extra=meta,
                        bbox=ocr_bbox,
                    ),
                )
        return candidates

    def _preprocess_for_mode(self, img: MatLike, mode: int) -> MatLike:
        """Apply specific OpenCV filters to optimize the image for Tesseract.

        Args:
            img: The source BGR image.
            mode: The integer ID representing the preprocessing pipeline.

        Returns:
            The processed grayscale or binary image.

        """
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if mode == OCR_MODE_OTSU:
            return cv2.threshold(
                grayscale,
                0,
                MAX_8BIT_VALUE,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]

        if mode == OCR_MODE_INVERTED_OTSU:
            return cv2.threshold(
                cv2.bitwise_not(grayscale),
                0,
                MAX_8BIT_VALUE,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]

        if mode == OCR_MODE_CUBIC_UPSCALE:
            return cv2.resize(
                grayscale,
                (0, 0),
                fx=OCR_GLOBAL_UPSCALE_FACTOR,
                fy=OCR_GLOBAL_UPSCALE_FACTOR,
                interpolation=INTERPOLATION_UP,
            )

        if mode in [OCR_MODE_TOPHAT_UPSCALE_A, OCR_MODE_TOPHAT_UPSCALE_B]:
            upscaled = cv2.resize(
                grayscale,
                (0, 0),
                fx=OCR_GLOBAL_UPSCALE_FACTOR,
                fy=OCR_GLOBAL_UPSCALE_FACTOR,
                interpolation=INTERPOLATION_UP,
            )
            morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                OCR_MORPH_KERNEL_SIZE,
            )
            top_hat = cv2.morphologyEx(upscaled, cv2.MORPH_TOPHAT, morph_kernel)
            return cv2.threshold(
                top_hat,
                0,
                MAX_8BIT_VALUE,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]

        return grayscale

    def recover_labels(
        self,
        img: MatLike,
        templates: list[Candidate],
        query: str,
        target_size: int,
        log_callback: Callable[..., Any],
    ) -> list[Candidate]:
        """Recover missing text labels by performing local OCR around icon hits.

        Args:
            img: The full screenshot/ROI.
            templates: Visual candidates found by the VisualProcessor.
            query: The label text expected near the icon.
            target_size: The pixel size of the icon for layout calculation.
            log_callback: Function to report recovery progress and show local ROIs.

        Returns:
            A list of candidates confirmed via local OCR recovery.

        """
        recovered_candidates: list[Candidate] = []
        query_lower = query.lower()

        for idx, template_hit in enumerate(templates):
            padding = int(target_size * RECOVERY_HORIZONTAL_PAD_FACTOR)
            y1 = template_hit.y + (target_size // 2) - RECOVERY_VERTICAL_OFFSET_PX
            y2 = template_hit.y + int(target_size * RECOVERY_VERTICAL_EXTEND_FACTOR)
            x1 = template_hit.x - (target_size // 2) - padding
            x2 = template_hit.x + (target_size // 2) + padding

            # Boundary clipping
            y1_clip, y2_clip = max(0, y1), min(img.shape[0], y2)
            x1_clip, x2_clip = max(0, x1), min(img.shape[1], x2)

            local_roi = img[y1_clip:y2_clip, x1_clip:x2_clip]

            if local_roi.size == 0:
                continue

            # Preprocessing for local recovery
            local_gray = cv2.cvtColor(local_roi, cv2.COLOR_BGR2GRAY)
            local_upscaled = cv2.resize(
                local_gray,
                (0, 0),
                fx=OCR_LOCAL_UPSCALE_FACTOR,
                fy=OCR_LOCAL_UPSCALE_FACTOR,
                interpolation=cv2.INTER_LINEAR,
            )
            _, thresholded = cv2.threshold(
                local_upscaled,
                0,
                MAX_8BIT_VALUE,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            if self.config.use_adaptive:
                thresholded = ImageUtils.enhance_contrast(thresholded)

            # UI Feedback: Show the tiny text-area being scanned
            log_callback(f"RECOVERY: Scanning Label ROI {idx + 1}...", thresholded)

            try:
                ocr_result = pytesseract.image_to_data(
                    thresholded,
                    output_type=pytesseract.Output.DICT,
                    config=f"--oem {OCR_ENGINE_MODE} --psm {PSM_SINGLE_LINE}",
                    timeout=2,
                )

                for i, text_token in enumerate(ocr_result.get("text", [])):
                    token_clean = str(text_token).strip().lower()
                    if len(token_clean) < OCR_MIN_TOKEN_LENGTH:
                        continue

                    similarity = SequenceMatcher(None, query_lower, token_clean).ratio()

                    # Scoring logic
                    if token_clean == query_lower:
                        recovery_score = OCR_EXACT_MATCH_SCORE - OCR_RECOVERY_PENALTY
                    elif query_lower in token_clean:
                        recovery_score = OCR_SUBSTRING_BASE_SCORE + (
                            OCR_SUBSTRING_MATCH_WEIGHT * similarity
                        )
                    else:
                        recovery_score = similarity * OCR_SIMILARITY_WEIGHT

                    if recovery_score >= OCR_RECOVERY_THRESHOLD:
                        # Fix: Ensure bbox is mapped back to global coordinates
                        recovered_candidates.append(
                            Candidate(
                                x=template_hit.x,
                                y=template_hit.y,
                                score=recovery_score,
                                method=DetectionMethod.OCR_RECOVERY,
                                bbox=template_hit.bbox,  # Use icon's bbox
                                extra={
                                    "original_method": template_hit.method,
                                    "recovered_text": token_clean,
                                    "ocr_conf": ocr_result["conf"][i],
                                },
                            ),
                        )
                        break
            except Exception as e:
                logger.warning("Recovery OCR failed for ROI %s: %s", idx + 1, e)

        return recovered_candidates
