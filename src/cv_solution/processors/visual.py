"""Provide template matching and feature-based (ORB) visual detection logic.

Execute high-performance visual searches across multiple color spaces,
edge maps, and scales to locate UI elements with pixel precision.
"""

import concurrent.futures
import logging
import time
from typing import TYPE_CHECKING, Any, ParamSpec

import cv2
import numpy as np

from cv_solution.constants import (
    BGR_TO_GRAY,
    BGR_TO_LAB,
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    MULTISCALE_FACTORS,
    ORB_DEFAULT_SCORE,
    ORB_MAX_FEATURES,
    ORB_MIN_MATCHES,
    ORB_NORM_TYPE,
    ORB_SAMPLE_POINTS,
    TPL_COLOR_THRESHOLD,
    TPL_EDGE_THRESHOLD,
    TPL_GRAY_THRESHOLD,
    TPL_LAB_THRESHOLD,
    TPL_MATCH_METHOD,
    TPL_MULTISCALE_THRESHOLD,
)
from cv_solution.models import Candidate, DetectionMethod, PerfStat

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cv2.typing import MatLike

    from cv_solution.models import GroundingConfig

logger = logging.getLogger(__name__)

P = ParamSpec("P")


class VisualProcessor:
    """Handle visual detection using template matching and keypoint features.

    Manage a parallelized suite of detection passes including color matching,
    edge analysis, and multiscale searches to find UI icons on a desktop.
    """

    def __init__(self, config: GroundingConfig) -> None:
        """Initialize the visual processor.

        Args:
            config: Configuration object containing enabled passes and core counts.

        """
        self.config = config
        self.last_stats: list[PerfStat] = []

    def run_all(
        self,
        roi: MatLike,
        icon_path: Path | None,
        target_size: int,
        log_callback: Callable[..., Any],
        should_abort: Callable[[], bool],
    ) -> list[Candidate]:
        """Execute enabled visual passes in parallel with real-time debug updates.

        This method coordinates the visual detection suite. It performs sequential
        pre-processing to provide a "live" feedback loop to the UI via log_callback
        before launching parallelized template matching and feature detection.

        Args:
            roi: The region of interest from the screen to search within.
            icon_path: Path to the template icon file.
            target_size: Expected width of the icon in the ROI.
            log_callback: Function to report progress, messages, and debug images.
            should_abort: Function to check if the process should stop early.

        Returns:
            A list of all detected candidates across all enabled passes. Returns
            an empty list if icon_path is None or if the process is aborted.

        Raises:
            Exception: Logs and continues if an individual detection pass fails.

        """
        if not icon_path or should_abort():
            return []

        hits: list[Candidate] = []
        self.last_stats = []

        # 1. Pre-process templates and primary ROI views
        # We do this sequentially to allow the UI to "flicker" through the logic modes
        tpl_bgr, tpl_mask, tpl_h = self._prep_template(icon_path, target_size)

        # UI Feedback: Initial BGR ROI
        log_callback("VISUAL: Initializing scan...", roi, progress=12)

        # UI Feedback: Grayscale Analysis
        roi_gray = cv2.cvtColor(roi, BGR_TO_GRAY)
        tpl_gray = cv2.cvtColor(tpl_bgr, BGR_TO_GRAY)
        log_callback("VISUAL: Analyzing Intensity (Gray)...", roi_gray, progress=15)
        time.sleep(0.02)  # Tiny pause to ensure the UI renders the frame

        # UI Feedback: Edge Detection (Canny)
        roi_edge = cv2.Canny(roi_gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        tpl_edge = cv2.Canny(tpl_gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        log_callback("VISUAL: Detecting Edges (Canny)...", roi_edge, progress=18)
        time.sleep(0.02)

        # UI Feedback: Color Space Analysis (LAB)
        roi_lab = None
        if self.config.use_lab:
            roi_lab = cv2.cvtColor(roi, BGR_TO_LAB)
            log_callback("VISUAL: Mapping Color Spaces (LAB)...", roi_lab, progress=20)
            time.sleep(0.02)

        # 2. Parallel Execution Suite
        # We reuse the pre-calculated maps (roi_edge, roi_gray) to save CPU cycles
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.num_cores,
        ) as executor:
            futures = []

            # Color Match (Uses original BGR)
            if self.config.use_color:
                futures.append(
                    executor.submit(
                        self._timed_run,
                        "Color Pass",
                        self._run_tpl_match,
                        roi,
                        tpl_bgr,
                        tpl_mask,
                        target_size,
                        tpl_h,
                        TPL_COLOR_THRESHOLD,
                        DetectionMethod.COLOR,
                    ),
                )

            # LAB Match (Uses pre-calculated LAB map)
            if self.config.use_lab and roi_lab is not None:
                futures.append(
                    executor.submit(
                        self._timed_run,
                        "LAB Pass",
                        self._run_tpl_match,
                        roi_lab,
                        cv2.cvtColor(tpl_bgr, BGR_TO_LAB),
                        None,
                        target_size,
                        tpl_h,
                        TPL_LAB_THRESHOLD,
                        DetectionMethod.LAB,
                    ),
                )

            # Edge Match (Uses pre-calculated Canny map)
            if self.config.use_edge:
                futures.append(
                    executor.submit(
                        self._timed_run,
                        "Edge Pass",
                        self._run_tpl_match,
                        roi_edge,
                        tpl_edge,
                        None,
                        target_size,
                        tpl_h,
                        TPL_EDGE_THRESHOLD,
                        DetectionMethod.EDGE,
                    ),
                )

            # Gray Match (Uses pre-calculated Gray map)
            if self.config.use_gray:
                futures.append(
                    executor.submit(
                        self._timed_run,
                        "Gray Pass",
                        self._run_tpl_match,
                        roi_gray,
                        tpl_gray,
                        None,
                        target_size,
                        tpl_h,
                        TPL_GRAY_THRESHOLD,
                        DetectionMethod.GRAY,
                    ),
                )

            # Feature-based ORB Match
            if self.config.use_orb:
                futures.append(
                    executor.submit(
                        self._timed_run,
                        "ORB Pass",
                        self._run_orb_pass,
                        roi,
                        icon_path,
                    ),
                )

            # Multiscale Analysis
            if self.config.use_multiscale:
                futures.extend(
                    [
                        executor.submit(
                            self._timed_run,
                            f"Scale {s}x",
                            self._run_scaled_pass,
                            roi,
                            icon_path,
                            target_size,
                            s,
                        )
                        for s in MULTISCALE_FACTORS
                    ],
                )

            # 3. Result Aggregation
            for future in concurrent.futures.as_completed(futures):
                if should_abort():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    result_hits, stat = future.result()
                    hits.extend(result_hits)
                    self.last_stats.append(stat)
                except Exception:
                    logger.exception("Visual detection pass failed")

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
            List of candidates found in this specific pass.

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
        start_time = time.time()
        result = func(*args, **kwargs)
        duration_ms = (time.time() - start_time) * 1000
        return result, PerfStat(name, duration_ms, len(result))

    def _prep_template(
        self,
        path: Path,
        target_width: int,
    ) -> tuple[MatLike, MatLike | None, int]:
        """Resize the template and extract an alpha mask if available.

        Args:
            path: Path to the template image.
            target_width: Desired width for resizing.

        Returns:
            A tuple of (resized_rgb_image, optional_mask, target_height).

        Raises:
            FileNotFoundError: If the template cannot be loaded.

        """
        template_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if template_raw is None:
            msg = f"Failed to load template at {path}"
            raise FileNotFoundError(msg)

        scale = target_width / template_raw.shape[1]
        target_height = int(template_raw.shape[0] * scale)

        mask = (
            cv2.resize(template_raw[:, :, 3], (target_width, target_height))
            if template_raw.shape[-1] == 4
            else None
        )
        rgb_image = cv2.resize(
            template_raw[:, :, 0:3] if mask is not None else template_raw,
            (target_width, target_height),
        )
        return rgb_image, mask, target_height

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
        locations = np.where(result_map >= threshold)
        candidates = []
        for pt in zip(*locations[::-1], strict=False):
            score = float(result_map[pt[1], pt[0]])
            candidates.append(
                Candidate(
                    x=int(pt[0] + width // 2),
                    y=int(pt[1] + height // 2),
                    score=score,
                    method=method,
                    bbox=(int(pt[0]), int(pt[1]), width, height),
                    extra=extra or {},
                ),
            )
        return candidates

    def _run_orb_pass(
        self,
        roi: MatLike,
        template_path: Path,
    ) -> list[Candidate]:
        """Detect clusters of matching keypoints using the ORB algorithm.

        Args:
            roi: The search image.
            template_path: Path to the original icon.

        Returns:
            A list containing a single Candidate if enough features match.

        """
        template = cv2.imread(str(template_path))
        if template is None:
            return []
        orb = cv2.ORB.create(nfeatures=ORB_MAX_FEATURES)
        _, descriptors1 = orb.detectAndCompute(template, None)
        keypoints2, descriptors2 = orb.detectAndCompute(roi, None)

        if descriptors1 is None or descriptors2 is None:
            return []

        matches = sorted(
            cv2.BFMatcher(ORB_NORM_TYPE, crossCheck=True).match(
                descriptors1,
                descriptors2,
            ),
            key=lambda x: x.distance,
        )

        if len(matches) > ORB_MIN_MATCHES:
            match_pts = np.array(
                [keypoints2[m.trainIdx].pt for m in matches[:ORB_SAMPLE_POINTS]],
            )
            center = np.mean(match_pts, axis=0)

            w, h = template.shape[1], template.shape[0]
            orb_bbox = (int(center[0] - w // 2), int(center[1] - h // 2), w, h)

            return [
                Candidate(
                    x=int(center[0]),
                    y=int(center[1]),
                    score=ORB_DEFAULT_SCORE,
                    method=DetectionMethod.ORB,
                    bbox=orb_bbox,
                ),
            ]
        return []

    def _run_scaled_pass(
        self,
        roi: MatLike,
        template_path: Path,
        target_width: int,
        scale_factor: float,
    ) -> list[Candidate]:
        """Execute a multiscale template matching pass.

        Args:
            roi: The search image.
            template_path: Path to the icon.
            target_width: Base width to scale from.
            scale_factor: The multiplier for resizing.

        Returns:
            List of detected candidates at this scale.

        """
        scaled_w = int(target_width * scale_factor)
        tpl, mask, scaled_h = self._prep_template(template_path, scaled_w)

        if scaled_h > roi.shape[0] or scaled_w > roi.shape[1]:
            return []

        return self._run_tpl_match(
            roi,
            tpl,
            mask,
            scaled_w,
            scaled_h,
            TPL_MULTISCALE_THRESHOLD,
            DetectionMethod.SCALE,
            extra={"scale_factor": scale_factor},
        )
