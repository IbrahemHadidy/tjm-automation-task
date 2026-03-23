"""Provide template matching logic.

Execute high-performance visual searches across multiple color spaces,
edge maps, and scales to locate UI elements with pixel precision.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ParamSpec

import cv2
import numpy as np

from cv_strategy.constants import (
    BGR_TO_GRAY,
    BGR_TO_LAB,
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    MULTISCALE_FACTORS,
    NMS_IOU_THRESHOLD,
    TPL_COLOR_THRESHOLD,
    TPL_EDGE_THRESHOLD,
    TPL_GRAY_THRESHOLD,
    TPL_LAB_THRESHOLD,
    TPL_MATCH_METHOD,
    TPL_MULTISCALE_THRESHOLD,
)
from cv_strategy.models import Candidate, DetectionMethod, PerfStat

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cv2.typing import MatLike

    from cv_strategy.models import GroundingConfig

logger = logging.getLogger(__name__)

P = ParamSpec("P")


@dataclass(slots=True)
class _DetectionJob:
    """Describe one visual detection pass."""

    name: str
    func: Callable[..., list[Candidate]]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(slots=True)
class _PreparedTemplate:
    """Hold all preprocessed template variants for a single target width."""

    bgr: MatLike
    gray: MatLike
    edge: MatLike
    lab: MatLike | None
    mask: MatLike | None
    height: int


class VisualProcessor:
    """Handle visual detection using template matching and keypoint features.

    The processor follows a three-stage pipeline:

    1. Prepare shared template and ROI representations once.
    2. Execute enabled detection passes in parallel.
    3. Fuse overlapping candidates with IoU-based non-maximum suppression.

    The implementation keeps the expensive work concentrated in OpenCV calls,
    caches repeated template preparation, and keeps the public control flow easy
    to follow for debugging and profiling.
    """

    def __init__(self, config: GroundingConfig) -> None:
        """Initialize the visual processor.

        Args:
            config: Configuration object containing enabled passes and core counts.

        """
        self.config = config
        self.last_stats: list[PerfStat] = []
        self._template_cache: dict[tuple[str, int], _PreparedTemplate] = {}
        self._orb_cache: dict[str, MatLike | None] = {}

    def _build_jobs(
        self,
        *,
        roi: MatLike,
        roi_gray: MatLike,
        roi_edge: MatLike,
        roi_lab: MatLike | None,
        icon_path: Path,
        base_target_size: int,
    ) -> list[_DetectionJob]:
        """Construct the list of enabled visual detection jobs.

        This method centralizes the creation of all detection passes based on the
        current configuration, including color, LAB, edge, grayscale, and
        multiscale template matching. Each job is represented as a _DetectionJob,
        containing the job name, the function to execute, and its arguments.

        Using this method ensures that the job setup is consistent and
        maintainable, and allows easy extension for new passes.

        Args:
            roi: The region of interest from the screen (BGR).
            roi_gray: Grayscale version of the ROI.
            roi_edge: Edge-detected version of the ROI.
            roi_lab: LAB color-space version of the ROI (or None if unused).
            icon_path: Path to the template icon file.
            base_target_size: Expected base width of the icon in the ROI before scaling.

        Returns:
            A list of _DetectionJob objects for all enabled detection passes.

        """
        jobs: list[_DetectionJob] = []

        def add_job(
            enabled: bool,  # noqa: FBT001
            name: str,
            func: Callable[..., list[Candidate]],
            args: tuple[Any, ...],
            kwargs: dict[str, Any] | None = None,
        ) -> None:
            if enabled:
                jobs.append(_DetectionJob(name, func, args, kwargs or {}))

        # Determine which scales to process
        scales = [1.0]
        if self.config.use_multiscale:
            for scale in MULTISCALE_FACTORS:
                if scale not in scales:
                    scales.append(scale)

        for scale in scales:
            scaled_w = int(base_target_size * scale)

            # Bounds check for width
            if scaled_w > roi.shape[1] or scaled_w <= 0:
                continue

            # Prepare templates for this specific scale
            prepared = self._prep_template(icon_path, scaled_w)

            # Bounds check for height
            if prepared.height > roi.shape[0]:
                continue

            extra = {"scale_factor": scale}
            scale_label = f" (Scale {scale}x)" if scale != 1.0 else ""

            add_job(
                self.config.use_color,
                f"Color Pass{scale_label}",
                self._run_tpl_match,
                (
                    roi,
                    prepared.bgr,
                    prepared.mask,
                    scaled_w,
                    prepared.height,
                    TPL_COLOR_THRESHOLD,
                    DetectionMethod.COLOR,
                    extra,
                ),
            )

            add_job(
                self.config.use_lab
                and roi_lab is not None
                and prepared.lab is not None,
                f"LAB Pass{scale_label}",
                self._run_tpl_match,
                (
                    roi_lab,
                    prepared.lab,
                    None,
                    scaled_w,
                    prepared.height,
                    TPL_LAB_THRESHOLD,
                    DetectionMethod.LAB,
                    extra,
                ),
            )

            add_job(
                self.config.use_edge,
                f"Edge Pass{scale_label}",
                self._run_tpl_match,
                (
                    roi_edge,
                    prepared.edge,
                    None,
                    scaled_w,
                    prepared.height,
                    TPL_EDGE_THRESHOLD,
                    DetectionMethod.EDGE,
                    extra,
                ),
            )

            add_job(
                self.config.use_gray,
                f"Gray Pass{scale_label}",
                self._run_tpl_match,
                (
                    roi_gray,
                    prepared.gray,
                    None,
                    scaled_w,
                    prepared.height,
                    TPL_GRAY_THRESHOLD,
                    DetectionMethod.GRAY,
                    extra,
                ),
            )

        return jobs

    def run_all(
        self,
        roi: MatLike,
        icon_path: Path | None,
        target_size: int,
        log_callback: Callable[..., Any],
        should_abort: Callable[[], bool],
    ) -> list[Candidate]:
        """Execute enabled visual passes with cached preprocessing and fusion.

        The pipeline is split into preprocessing and detection passes.
        Template preparation and ROI conversions are done once.

        Args:
            roi: The region of interest from the screen to search within.
            icon_path: Path to the template icon file.
            target_size: Expected width of the icon in the ROI.
            log_callback: Function to report progress messages (text only).
            should_abort: Function to check if the process should stop early.

        Returns:
            A list of detected candidates across all enabled passes. Empty if
            icon_path is None or aborted.

        """
        if not icon_path or should_abort():
            return []

        self.last_stats = []
        hits: list[Candidate] = []

        # Process ROI once
        roi_gray = cv2.cvtColor(roi, BGR_TO_GRAY)
        roi_edge = cv2.Canny(roi_gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        roi_lab = cv2.cvtColor(roi, BGR_TO_LAB) if self.config.use_lab else None

        # Build jobs across all enabled passes and scales
        jobs = self._build_jobs(
            roi=roi,
            roi_gray=roi_gray,
            roi_edge=roi_edge,
            roi_lab=roi_lab,
            icon_path=icon_path,
            base_target_size=target_size,
        )

        if not jobs:
            log_callback("Visual Suite Completed", progress=40)
            return []

        max_workers = min(self.config.num_cores, len(jobs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map: dict[
                concurrent.futures.Future[tuple[list[Candidate], PerfStat]],
                str,
            ] = {
                executor.submit(
                    self._timed_run,
                    job.name,
                    job.func,
                    *job.args,
                    **job.kwargs,
                ): job.name
                for job in jobs
            }

            for future in concurrent.futures.as_completed(future_map):
                if should_abort():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                try:
                    result_hits, stat = future.result()
                except Exception:
                    logger.exception(
                        "Visual detection pass failed: %s",
                        future_map[future],
                    )
                    continue

                hits.extend(result_hits)
                self.last_stats.append(stat)

        hits = self._fuse_candidates(hits)
        log_callback("Visual Suite Completed", progress=40)
        return hits

    def _run_tpl_match(
        self,
        roi_proc: MatLike,
        tpl_proc: MatLike,
        mask: MatLike | None,
        w: int,
        h: int,
        threshold: float,
        method: DetectionMethod,
        extra: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        """Execute a generic template-matching pass for visual detection.

        Args:
            roi_proc: Pre-processed ROI image.
            tpl_proc: Pre-processed template image.
            mask: Optional alpha mask for transparency.
            w: Width of the template.
            h: Height of the template.
            threshold: Confidence threshold for detection.
            method: The DetectionMethod enum member.
            extra: Optional metadata for the Candidate.

        Returns:
            A list of candidates found in this specific pass.

        """
        result_map = cv2.matchTemplate(roi_proc, tpl_proc, TPL_MATCH_METHOD, mask=mask)
        return self._extract_locations(result_map, threshold, w, h, method, extra)

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

    def _prep_template(
        self,
        path: Path,
        target_width: int,
    ) -> _PreparedTemplate:
        """Load, resize, and cache all template variants for one target width.

        Args:
            path: Path to the template image.
            target_width: Desired width for resizing.

        Returns:
            A prepared template bundle containing resized BGR, grayscale, edge,
            optional LAB, optional alpha mask, and the resized height.

        Raises:
            FileNotFoundError: If the template cannot be loaded.

        """
        cache_key = (str(path), target_width)
        cached = self._template_cache.get(cache_key)
        if cached is not None:
            return cached

        template_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if template_raw is None:
            msg = f"Failed to load template at {path}"
            raise FileNotFoundError(msg)

        # OpenCV may return grayscale, BGR, or BGRA images depending on the file.
        # Normalize the template into a predictable BGR pipeline before caching.
        if template_raw.ndim == 2:
            base_bgr = cv2.cvtColor(template_raw, cv2.COLOR_GRAY2BGR)
            mask = None
        else:
            has_alpha = template_raw.shape[2] == 4
            base_bgr = template_raw[:, :, 0:3] if has_alpha else template_raw
            mask = (
                cv2.resize(
                    template_raw[:, :, 3],
                    (
                        target_width,
                        int(
                            template_raw.shape[0]
                            * target_width
                            / template_raw.shape[1],
                        ),
                    ),
                )
                if has_alpha
                else None
            )

        scale = target_width / base_bgr.shape[1]
        target_height = int(base_bgr.shape[0] * scale)
        resize_size = (target_width, target_height)

        bgr = cv2.resize(base_bgr, resize_size)
        if mask is not None:
            mask = cv2.resize(mask, resize_size)
        gray = cv2.cvtColor(bgr, BGR_TO_GRAY)
        edge = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        lab = cv2.cvtColor(bgr, BGR_TO_LAB) if self.config.use_lab else None

        prepared = _PreparedTemplate(
            bgr=bgr,
            gray=gray,
            edge=edge,
            lab=lab,
            mask=mask,
            height=target_height,
        )
        self._template_cache[cache_key] = prepared
        return prepared

    def _extract_locations(
        self,
        result_map: MatLike,
        threshold: float,
        width: int,
        height: int,
        method: DetectionMethod,
        extra: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        """Convert a matchTemplate result map into a list of Candidate objects.

        Args:
            result_map: The cross-correlation map from OpenCV.
            threshold: Minimum score to consider a hit.
            width: Width of the detected bounding box.
            height: Height of the detected bounding box.
            method: The strategy used for this detection.
            extra: Optional metadata dictionary.

        Returns:
            A list of validated Candidate objects.

        """
        matches = np.argwhere(result_map >= threshold)
        if matches.size == 0:
            return []

        candidates: list[Candidate] = []
        for y, x in matches:
            score = float(result_map[y, x])
            if not np.isfinite(score):
                continue

            candidate = Candidate(
                x=int(x + width // 2),
                y=int(y + height // 2),
                score=score,
                method=method,
                bbox=(int(x), int(y), width, height),
                extra=extra or {},
            )
            candidates.append(candidate)

        return candidates

    def _run_scaled_pass(
        self,
        roi: MatLike,
        template_path: Path,
        target_width: int,
        scale_factor: float,
    ) -> list[Candidate]:
        """Execute a multiscale template-matching pass.

        Args:
            roi: The search image.
            template_path: Path to the icon.
            target_width: Base width to scale from.
            scale_factor: The multiplier for resizing.

        Returns:
            List of detected candidates at this scale.

        """
        scaled_w = int(target_width * scale_factor)
        prepared = self._prep_template(template_path, scaled_w)

        if prepared.height > roi.shape[0] or scaled_w > roi.shape[1]:
            return []

        return self._run_tpl_match(
            roi,
            prepared.bgr,
            prepared.mask,
            scaled_w,
            prepared.height,
            TPL_MULTISCALE_THRESHOLD,
            DetectionMethod.SCALE,
            extra={"scale_factor": scale_factor},
        )

    def _fuse_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        """Merge near-duplicate candidates from multiple passes.

        The fusion step removes excessive overlap between detections that point to
        the same visual object but were produced by different passes or scales.

        This implementation uses greedy IoU-based non-maximum suppression instead
        of center-distance heuristics. IoU is more geometrically meaningful when
        the same icon is detected with slightly different box sizes or offsets.

        Args:
            candidates: Raw candidates emitted by the enabled passes.

        Returns:
            A deduplicated list of candidates sorted by descending score.

        """
        if len(candidates) < 2:
            return candidates

        spatial_candidates = [candidate for candidate in candidates if candidate.bbox]
        non_spatial_candidates = [
            candidate for candidate in candidates if not candidate.bbox
        ]

        fused_spatial = self._nms_candidates(spatial_candidates, NMS_IOU_THRESHOLD)
        fused = fused_spatial + sorted(
            non_spatial_candidates,
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        return sorted(fused, key=lambda candidate: candidate.score, reverse=True)

    def _nms_candidates(
        self,
        candidates: list[Candidate],
        iou_threshold: float,
    ) -> list[Candidate]:
        """Apply greedy IoU-based non-maximum suppression.

        Args:
            candidates: Spatial candidates with bounding boxes.
            iou_threshold: Maximum intersection-over-union allowed before a box is
                considered a duplicate of a higher-scoring box.

        Returns:
            The highest-scoring non-overlapping candidates.

        """
        if len(candidates) < 2:
            return candidates

        ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
        kept: list[Candidate] = []

        for candidate in ranked:
            if candidate.bbox is None:
                continue

            if all(
                self._box_iou(candidate.bbox, other.bbox) <= iou_threshold
                for other in kept
                if other.bbox is not None
            ):
                kept.append(candidate)

        return kept

    @staticmethod
    def _box_iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """Calculate intersection-over-union for two axis-aligned boxes.

        Args:
            box_a: Bounding box in ``(x, y, w, h)`` format.
            box_b: Bounding box in ``(x, y, w, h)`` format.

        Returns:
            IoU in the ``0.0`` to ``1.0`` range.

        """
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)

        inter_w = max(0, right - left)
        inter_h = max(0, bottom - top)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0

        return inter_area / union_area
