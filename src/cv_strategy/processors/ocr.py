"""Provide OCR preprocessing, recognition, and recovery logic for UI detection.

Execute high-throughput OCR passes across multiple preprocessing variants to
extract text labels with stable bounding boxes and confidence scores.

The pipeline mirrors the visual processor design:

1. Prepare shared ROI variants once.
2. Execute enabled OCR passes in parallel.
3. Return all query-matching candidates, preserving distinct text hits.

This structure keeps calibration points explicit while avoiding repeated image
conversion, repeated thresholding, and repeated OCR invocations on identical
inputs.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, ParamSpec

import cv2
import pytesseract
from pytesseract import Output

from cv_strategy.constants import (
    BGR_TO_GRAY,
    INTERPOLATION_LOCAL,
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
    OCR_MORPH_KERNEL_SIZE,
    OCR_RECOVERY_PENALTY,
    OCR_RECOVERY_THRESHOLD,
    OCR_SIMILARITY_WEIGHT,
    OCR_SUBSTRING_BASE_SCORE,
    OCR_SUBSTRING_MATCH_WEIGHT,
    PSM_SINGLE_LINE,
    RECOVERY_HORIZONTAL_PAD_FACTOR,
    RECOVERY_QUEUE_LIMIT,
    RECOVERY_VERTICAL_EXTEND_FACTOR,
    RECOVERY_VERTICAL_OFFSET_PX,
    OCRPreprocessingMode,
)
from cv_strategy.models import Candidate, DetectionMethod, PerfStat
from cv_strategy.utils import ImageUtils

if TYPE_CHECKING:
    from collections.abc import Callable

    from cv2.typing import MatLike

    from cv_strategy.models import GroundingConfig

logger = logging.getLogger(__name__)

P = ParamSpec("P")


@dataclass(slots=True)
class _PreparedOCRImage:
    """Hold preprocessed OCR variants derived from a single ROI.

    Attributes:
        gray: Grayscale ROI used as the base OCR input.
        inverted: Inverted grayscale ROI for light-text-on-dark backgrounds.
        otsu: Otsu-thresholded ROI for higher-contrast segmentation.
        morph: Morphologically enhanced ROI for isolating text strokes.
        upscale: Upscaled grayscale ROI for small glyph recovery.

    """

    gray: MatLike
    inverted: MatLike
    otsu: MatLike
    morph: MatLike
    upscale: MatLike


@dataclass(slots=True)
class _OCRJob:
    """Describe a single OCR pass submission."""

    name: str
    image: MatLike
    query: str
    psm: int
    mode: DetectionMethod
    scale: float
    offset_x: int
    offset_y: int
    extra: dict[str, Any]


class OCRProcessor:
    """Handle OCR detection, query matching, and local label recovery.

    The processor is designed for UI text extraction rather than document OCR.
    It favors a small set of high-signal preprocessing variants, parallel
    execution across those variants, and query-aware candidate extraction that
    preserves valid alternate matches instead of collapsing them into a single
    winner too early.
    """

    def __init__(self, tesseract_path: str, config: GroundingConfig) -> None:
        """Initialize the OCR processor.

        Args:
            tesseract_path: Absolute path to the Tesseract executable.
            config: Configuration object containing OCR behavior and core counts.

        """
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.config = config
        self.last_stats: list[PerfStat] = []

    def search_global(
        self,
        roi: MatLike,
        query: str,
        log_callback: Callable[..., Any],
    ) -> list[Candidate]:
        """Run multiple preprocessing modes to find text across the image.

        Args:
            roi: The image or region to search for text.
            query: The text string to locate.
            log_callback: Function to report progress and show debug images.

        Returns:
            A list of OCR candidates where the recognized text matches the query.

        Notes:
            This method intentionally returns all matching candidates rather than
            applying aggressive fusion. Global OCR can legitimately find the same
            query in multiple positions or preprocessing variants, and collapsing
            them too early hides useful evidence.

        """
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []

        self.last_stats = []
        jobs = self._build_global_jobs(roi, normalized_query, log_callback)
        if not jobs:
            return []

        hits: list[Candidate] = []
        max_workers = min(self.config.num_cores, len(jobs))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map: dict[
                concurrent.futures.Future[tuple[list[Candidate], PerfStat]],
                str,
            ] = {}

            for job in jobs:
                future = executor.submit(
                    self._timed_run,
                    job.name,
                    self.run_ocr_pass,
                    job,
                )
                future_map[future] = job.name

            for future in concurrent.futures.as_completed(future_map):
                try:
                    result_hits, stat = future.result()
                except Exception:
                    logger.exception("OCR global pass failed: %s", future_map[future])
                    continue

                hits.extend(result_hits)
                self.last_stats.append(stat)

        return hits

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
            img: The full screenshot or ROI.
            templates: Visual candidates found by the visual processor.
            query: The label text expected near the icon.
            target_size: The pixel size of the icon for layout calculation.
            log_callback: Function to report recovery progress and show local ROIs.

        Returns:
            A list of candidates confirmed via targeted OCR recovery.

        Notes:
            Recovery uses stricter text verification than the global pass and keeps
            results anchored to the icon that triggered the local crop.

        """
        normalized_query = query.strip().lower()
        if not templates or not normalized_query:
            return []

        queue = templates[: min(RECOVERY_QUEUE_LIMIT, len(templates))]
        log_callback(
            f"RECOVERY: Verifying {len(queue)} anchors",
            progress=82,
        )

        self.last_stats = []
        jobs = self._build_recovery_jobs(
            img,
            queue,
            normalized_query,
            target_size,
            log_callback,
        )
        if not jobs:
            return []

        hits: list[Candidate] = []
        max_workers = min(self.config.num_cores, len(jobs))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map: dict[
                concurrent.futures.Future[tuple[list[Candidate], PerfStat]],
                str,
            ] = {}

            for job in jobs:
                future = executor.submit(
                    self._timed_run,
                    job.name,
                    self.run_ocr_pass,
                    job,
                )
                future_map[future] = job.name

            for future in concurrent.futures.as_completed(future_map):
                try:
                    result_hits, stat = future.result()
                except Exception:
                    logger.exception("OCR recovery pass failed: %s", future_map[future])
                    continue

                hits.extend(result_hits)
                self.last_stats.append(stat)

        return hits

    def run_ocr_pass(self, job: _OCRJob) -> list[Candidate]:
        """Run Tesseract OCR on a single preprocessed image variant.

        Args:
            job: OCR job descriptor containing image, query, mode, and metadata.

        Returns:
            A list of OCR candidates extracted from the image.

        Notes:
            This pass keeps the OCR engine call isolated so different preprocessing
            strategies can be benchmarked and tuned independently.

        """
        if job.mode not in (DetectionMethod.OCR_GLOBAL, DetectionMethod.OCR_RECOVERY):
            msg = "mode must be OCR_GLOBAL or OCR_RECOVERY"
            raise ValueError(msg)

        config = f"--oem {OCR_ENGINE_MODE} --psm {job.psm}"
        data = pytesseract.image_to_data(
            job.image,
            output_type=Output.DICT,
            config=config,
            lang=self.config.ocr_lang,
        )
        return self._parse_tesseract_data(
            data=data,
            query=job.query,
            scale=job.scale,
            method=job.mode,
            extra=job.extra,
            offset_x=job.offset_x,
            offset_y=job.offset_y,
        )

    def _build_global_jobs(
        self,
        roi: MatLike,
        query: str,
        log_callback: Callable[..., Any],
    ) -> list[_OCRJob]:
        """Build OCR jobs for the global search pass.

        Args:
            roi: The image or region to search for text.
            query: Lower-cased text query.
            log_callback: Function used to emit progress images.

        Returns:
            A list of OCR job descriptors ready for threaded execution.

        """
        jobs: list[_OCRJob] = []
        upscale_modes = {
            OCRPreprocessingMode.CUBIC_UPSCALE,
            OCRPreprocessingMode.TOPHAT_UPSCALE_A,
            OCRPreprocessingMode.TOPHAT_UPSCALE_B,
        }

        for mode_id in OCR_GLOBAL_SEARCH_MODES:
            processed_image = self._preprocess_for_mode(roi, mode_id)
            if self.config.use_adaptive:
                processed_image = ImageUtils.enhance_contrast(processed_image)

            log_callback(f"OCR: Scanning Mode {mode_id.name}...", processed_image)

            psm_mode = (
                PSM_SINGLE_LINE
                if mode_id == OCRPreprocessingMode.TOPHAT_UPSCALE_A
                else self.config.psm
            )
            scale_factor = (
                OCR_GLOBAL_UPSCALE_FACTOR if mode_id in upscale_modes else 1.0
            )

            jobs.append(
                _OCRJob(
                    name=f"OCR {mode_id.name}",
                    image=processed_image,
                    query=query,
                    psm=psm_mode,
                    mode=DetectionMethod.OCR_GLOBAL,
                    scale=scale_factor,
                    offset_x=0,
                    offset_y=0,
                    extra={
                        "ocr_mode": mode_id.name.lower(),
                        "upscaled": scale_factor > 1.0,
                    },
                ),
            )

        return jobs

    def _build_recovery_jobs(
        self,
        img: MatLike,
        templates: list[Candidate],
        query: str,
        target_size: int,
        log_callback: Callable[..., Any],
    ) -> list[_OCRJob]:
        """Build OCR jobs for the local recovery pass.

        Args:
            img: The full screenshot or ROI.
            templates: Visual anchors used to define recovery ROIs.
            query: Lower-cased text query.
            target_size: Base size for the crop geometry.
            log_callback: Function used to emit progress images.

        Returns:
            A list of OCR job descriptors ready for threaded execution.

        """
        jobs: list[_OCRJob] = []

        for idx, template_hit in enumerate(templates):
            crop_bounds = self._compute_recovery_bounds(img, template_hit, target_size)
            if crop_bounds is None:
                continue

            x1, y1, x2, y2 = crop_bounds
            local_roi = img[y1:y2, x1:x2]
            if local_roi.size == 0:
                continue

            local_gray = cv2.cvtColor(local_roi, BGR_TO_GRAY)
            local_upscaled = cv2.resize(
                local_gray,
                (0, 0),
                fx=OCR_LOCAL_UPSCALE_FACTOR,
                fy=OCR_LOCAL_UPSCALE_FACTOR,
                interpolation=INTERPOLATION_LOCAL,
            )
            _, thresholded = cv2.threshold(
                local_upscaled,
                0,
                MAX_8BIT_VALUE,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            if self.config.use_adaptive:
                thresholded = ImageUtils.enhance_contrast(thresholded)

            log_callback(f"RECOVERY: Scanning Label ROI {idx + 1}...", thresholded)

            jobs.append(
                _OCRJob(
                    name=f"OCR Recovery {idx + 1}",
                    image=thresholded,
                    query=query,
                    psm=PSM_SINGLE_LINE,
                    mode=DetectionMethod.OCR_RECOVERY,
                    scale=OCR_LOCAL_UPSCALE_FACTOR,
                    offset_x=x1,
                    offset_y=y1,
                    extra={
                        "anchor_index": idx,
                        "anchor_method": template_hit.method.value
                        if hasattr(template_hit.method, "value")
                        else str(template_hit.method),
                    },
                ),
            )

        return jobs

    def _compute_recovery_bounds(
        self,
        img: MatLike,
        template_hit: Candidate,
        target_size: int,
    ) -> tuple[int, int, int, int] | None:
        """Compute the local OCR crop around an icon anchor.

        Args:
            img: The full screenshot or ROI.
            template_hit: The visual anchor candidate.
            target_size: Base size for the crop geometry.

        Returns:
            A clipped crop box in ``(x1, y1, x2, y2)`` format or ``None`` if the
            candidate does not provide enough geometry.

        Notes:
            The crop is asymmetric in the vertical axis because desktop labels are
            typically located below the icon rather than centered on it.

        """
        if template_hit.bbox is None:
            return None

        x, y, w, h = template_hit.bbox
        center_x = x + (w // 2)
        center_y = y + (h // 2)

        padding = int(target_size * RECOVERY_HORIZONTAL_PAD_FACTOR)
        x1 = center_x - (target_size // 2) - padding
        x2 = center_x + (target_size // 2) + padding
        y1 = center_y - (target_size // 2) - RECOVERY_VERTICAL_OFFSET_PX
        y2 = center_y + int(target_size * RECOVERY_VERTICAL_EXTEND_FACTOR)

        x1_clip = max(0, x1)
        y1_clip = max(0, y1)
        x2_clip = min(img.shape[1], x2)
        y2_clip = min(img.shape[0], y2)

        if x2_clip <= x1_clip or y2_clip <= y1_clip:
            return None

        return x1_clip, y1_clip, x2_clip, y2_clip

    def _preprocess_for_mode(self, img: MatLike, mode: OCRPreprocessingMode) -> MatLike:
        """Apply preprocessing tailored to a specific OCR search mode.

        Args:
            img: The source BGR image.
            mode: The preprocessing mode to execute.

        Returns:
            A grayscale or binary image prepared for Tesseract.

        """
        grayscale = cv2.cvtColor(img, BGR_TO_GRAY)

        if mode == OCRPreprocessingMode.OTSU:
            return cv2.threshold(
                grayscale,
                0,
                MAX_8BIT_VALUE,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]

        if mode == OCRPreprocessingMode.INVERTED_OTSU:
            return cv2.threshold(
                cv2.bitwise_not(grayscale),
                0,
                MAX_8BIT_VALUE,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]

        if mode == OCRPreprocessingMode.CUBIC_UPSCALE:
            return cv2.resize(
                grayscale,
                (0, 0),
                fx=OCR_GLOBAL_UPSCALE_FACTOR,
                fy=OCR_GLOBAL_UPSCALE_FACTOR,
                interpolation=INTERPOLATION_UP,
            )

        if mode in (
            OCRPreprocessingMode.TOPHAT_UPSCALE_A,
            OCRPreprocessingMode.TOPHAT_UPSCALE_B,
        ):
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

    def _parse_tesseract_data(
        self,
        data: dict[str, Any],
        query: str,
        scale: float,
        method: DetectionMethod,
        extra: dict[str, Any] | None = None,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[Candidate]:
        """Extract query-matching candidates from Tesseract's dictionary output.

        Args:
            data: Dictionary returned by ``pytesseract.image_to_data``.
            query: The target text to match against.
            scale: The scaling factor applied to the image before OCR.
            method: The detection strategy used.
            extra: Optional metadata to attach to candidates.
            offset_x: Horizontal offset of the OCR crop in the parent image.
            offset_y: Vertical offset of the OCR crop in the parent image.

        Returns:
            List of parsed Candidate objects.

        Notes:
            The parser keeps exact matches, substring matches, and fuzzy matches,
            which preserves the broader result set expected by the original OCR
            pipeline while still rejecting weak OCR noise.

        """
        candidates: list[Candidate] = []
        query_lower = query.strip().lower()

        texts = data.get("text", [])
        confs = data.get("conf", [])
        lefts = data.get("left", [])
        tops = data.get("top", [])
        widths = data.get("width", [])
        heights = data.get("height", [])

        for i, text in enumerate(texts):
            found_text = str(text).strip().lower()
            if len(found_text) < OCR_MIN_TOKEN_LENGTH:
                continue

            try:
                confidence = float(confs[i])
                left = float(lefts[i])
                top = float(tops[i])
                width = float(widths[i])
                height = float(heights[i])
            except IndexError, KeyError, TypeError, ValueError:
                continue

            if confidence < OCR_MIN_CONFIDENCE or width <= 0 or height <= 0:
                continue

            similarity = SequenceMatcher(None, query_lower, found_text).ratio()
            matches_query = (
                found_text == query_lower
                or query_lower in found_text
                or similarity > OCR_FUZZY_MATCH_THRESHOLD
            )
            if not matches_query:
                continue

            if found_text == query_lower:
                score = OCR_EXACT_MATCH_SCORE
                if method == DetectionMethod.OCR_RECOVERY:
                    score = max(0.0, score - OCR_RECOVERY_PENALTY)
            elif query_lower in found_text:
                score = OCR_SUBSTRING_BASE_SCORE + (
                    OCR_SUBSTRING_MATCH_WEIGHT * similarity
                )
            else:
                score = similarity * OCR_SIMILARITY_WEIGHT

            if (
                method == DetectionMethod.OCR_RECOVERY
                and score < OCR_RECOVERY_THRESHOLD
            ):
                continue

            global_left = int((left / scale) + offset_x)
            global_top = int((top / scale) + offset_y)
            global_width = int(width / scale)
            global_height = int(height / scale)
            center_x = int(global_left + (global_width // 2))
            center_y = int(global_top + (global_height // 2))

            meta = extra.copy() if extra else {}
            meta.update(
                {
                    "text": found_text,
                    "conf": confidence,
                    "similarity": similarity,
                    "text_bbox": (global_left, global_top, global_width, global_height),
                },
            )

            candidates.append(
                Candidate(
                    x=center_x,
                    y=center_y,
                    score=score,
                    method=method,
                    bbox=(global_left, global_top, global_width, global_height),
                    extra=meta,
                ),
            )

        return candidates

    def _timed_run(
        self,
        name: str,
        func: Callable[P, list[Candidate]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[list[Candidate], PerfStat]:
        """Measure the execution time of a specific detection pass.

        Args:
            name: Label for the performance statistic.
            func: The detection function to execute.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            A tuple containing the list of found candidates and a PerfStat object.

        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000
        return result, PerfStat(name, duration_ms, len(result))
