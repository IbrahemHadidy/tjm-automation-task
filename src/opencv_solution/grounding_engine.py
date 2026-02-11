"""DesktopGroundingEngine.

Provides computer vision and OCR-based desktop UI element grounding using
a multi-pass probabilistic detection pipeline.

Design Philosophy
-----------------
This engine does not rely on a single detection method. Instead, it combines
multiple weak signals to improve robustness in real-world desktop environments.

Detection strategies include:
- Template matching (color, grayscale, edge, LAB space)
- Feature matching using ORB keypoints
- Multi-pass OCR text recognition using Tesseract
- Local recovery OCR around high-confidence visual matches

Results from these detectors are fused using spatial proximity heuristics
and confidence scoring.

Coordinate System
-----------------
All detections are represented using **center coordinates** to simplify
spatial fusion and duplicate removal.

Heuristic Nature
----------------
Many thresholds and constants are empirically tuned for typical Windows
desktop environments and should be adjusted for different screen layouts.
"""

import ctypes

# High-DPI Scaling Configuration
try:
    ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
except Exception:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

import concurrent.futures
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, ParamSpec

import cv2
import numpy as np
import pytesseract

from opencv_solution.constants import (
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    DEFAULT_ICON_SIZE,
    FINAL_DEDUP_RADIUS_FACTOR,
    FUSION_DISTANCE_FACTOR,
    ICON_MAX_WIDTH,
    ICON_MIN_WIDTH,
    MAX_TEMPLATE_HITS,
    MULTISCALE_FACTORS,
    NMS_RADIUS_FACTOR,
    OCR_MIN_CONFIDENCE,
    OCR_MIN_TOKEN_LENGTH,
    OCR_RECOVERY_THRESHOLD,
    ORB_MIN_MATCHES,
    ORB_SAMPLE_POINTS,
    RECOVERY_HORIZONTAL_PAD_FACTOR,
    RECOVERY_QUEUE_LIMIT,
    RECOVERY_VERTICAL_EXTEND_FACTOR,
    TASKBAR_HEIGHT_PX,
    TOKEN_DEDUP_RADIUS_PX,
    TPL_COLOR_THRESHOLD,
    TPL_EDGE_THRESHOLD,
    TPL_GRAY_THRESHOLD,
    TPL_LAB_THRESHOLD,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cv2.typing import MatLike


P = ParamSpec("P")

# A callback signature used by the engine to emit logs and intermediate visualizations.
type LogCallback = Callable[[str, str, int | None], None]


@dataclass
class Candidate:
    """Represent a potential UI element location found by the engine."""

    x: int
    y: int
    score: float
    method: str
    img_score: float = 0.0
    txt_score: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    geometry_score: float = 1.0


@dataclass
class PerfStat:
    """Store performance metrics for a specific detection pass."""

    name: str
    duration_ms: float
    items_found: int


class DesktopGroundingEngine:
    """Execute multi-modal searches for UI elements using OpenCV and Tesseract.

    The engine intentionally separates concerns:
    - many independent detection passes produce Candidate lists,
    - results are fused later with spatial heuristics and NMS,
    - a recovery OCR pass attempts to read labels close to template matches.
    """

    def __init__(self, tesseract_path: str) -> None:
        """Initialize the engine with the provided Tesseract executable path.

        Args:
            tesseract_path: Path to the tesseract executable (pytesseract.pytesseract.tesseract_cmd).

        """
        self.tesseract_path: str = tesseract_path
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        # Latest debug frame (for UI display) is stored here when logging frames.
        self.debug_frame: MatLike | None = None

        # Perf stats collected from timed wrappers for each pass.
        self.perf_stats: list[PerfStat] = []

        # Abort hook: external caller can set this to stop long-running ops.
        # Should be a callable that returns True when the engine should abort.
        self.should_abort: Callable[[], bool] = lambda: False

    def select_best_candidate(
        self,
        candidates: list[Candidate],
        priority: str = "fusion",
    ) -> Candidate | None:
        """Select the highest-confidence candidate using a priority strategy.

        Priority Strategies
        -------------------
        fusion:
        Prefer candidates produced by both visual and OCR detection.
        This typically represents the strongest semantic + visual agreement.

        text:
        Prefer OCR-derived detections, useful when text labels are more reliable
        than icon appearance.

        default fallback:
        Select the candidate with the highest raw confidence score.

        Returns
        -------
        Candidate | None
        Best matching detection or None if no candidates exist.

        """
        if not candidates:
            return None

        # Fusion preference: fused_match indicates matched by both template and OCR.
        if priority == "fusion":
            fused = [c for c in candidates if c.method == "fused_match"]
            if fused:
                return max(fused, key=lambda c: c.score)

        # Text preference: useful when label text is more reliable than icon visuals.
        if priority == "text":
            ocr_cands = [c for c in candidates if "ocr" in c.method]
            if ocr_cands:
                return max(ocr_cands, key=lambda c: c.score)

        # Default fallback (no preference): pick the absolute highest score.
        return max(candidates, key=lambda c: c.score)

    def locate_elements(
        self,
        screenshot_path: Path,
        icon_image: Path | None,
        text_query: str,
        threshold: float = 0.5,
        psm: int = 11,
        config: dict[str, Any] | None = None,
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Locate UI elements on screen by executing a multi-stage vision pipeline.

        Pipeline Flow
        -------------
        1. Load screenshot and remove UI chrome regions (e.g., taskbar)
        2. Estimate desktop icon size for adaptive template scaling
        3. Execute multiple template matching passes in parallel:
        - Color matching
        - CIELAB color space matching
        - Edge map matching
        - Grayscale matching
        - ORB feature matching
        4. Run global OCR sweeps for semantic text recognition
        5. Perform targeted OCR recovery around template detections
        6. Fuse visual and text detections using spatial heuristics
        7. Apply non-maximum suppression and score filtering

        Parameters
        ----------
        screenshot_path : Path
        Path to screenshot image file.

        icon_image : Path | None
        Reference template icon to search for.

        text_query : str
        Text label expected to appear near target UI element.

        threshold : float
        Minimum confidence score required to keep a detection.

        psm : int
        Tesseract Page Segmentation Mode controlling text layout assumptions.

        config : dict | None
        Optional configuration flags controlling detection passes.

        Returns
        -------
        list[Candidate]
        Ranked list of detected UI element candidates.

        """
        self.perf_stats = []
        safe_config: dict[str, Any] = config or {}
        t0 = time.time()

        full_img = self._load_and_preprocess_screenshot(screenshot_path)
        img = full_img.copy()
        desktop_roi = self._crop_desktop_roi(img)
        self._log("INIT: Screenshot Loaded", desktop_roi, callback, progress=5)

        if self.should_abort():
            return []

        # Heuristic: detect icon size to scale template appropriately.
        target_size = self._detect_desktop_icon_size(
            desktop_roi,
        )

        template_hits = self._run_template_passes(
            desktop_roi,
            icon_image,
            target_size,
            safe_config,
            callback,
        )

        ocr_hits = self._run_ocr(
            desktop_roi,
            text_query,
            psm,
            safe_config,
            callback,
        )

        # Non-maximum suppression to reduce dense template detections,
        # then optionally validate geometry against the template aspect ratio.
        templates = self._validate_templates(
            icon_image,
            self._non_max_suppression(template_hits, target_size),
        )

        # Try to recover labels locally near template matches (if OCR missed them globally).
        ocr_hits.extend(
            self._targeted_recovery(
                desktop_roi,
                templates,
                text_query,
                target_size,
                safe_config,
                callback,
            ),
        )

        # Fuse template and OCR hits, then apply thresholds and final NMS.
        final_candidates = self._finalize_results(
            templates,
            ocr_hits,
            target_size,
            threshold,
        )

        self._report_results(full_img, final_candidates, t0, callback)
        return final_candidates

    def _log(
        self,
        msg: str,
        frame: MatLike | None = None,
        callback: LogCallback | None = None,
        lvl: str = "INFO",
        progress: int | None = None,
    ) -> None:
        """Log a message and optionally store a debug frame for UI display.

        If `frame` is grayscale, convert it to BGR for downstream visualization.
        """
        if self.should_abort():
            return
        if frame is not None:
            self.debug_frame = (
                cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if len(frame.shape) == 2
                else frame.copy()
            )
        if callback:
            callback(msg, lvl, progress)

    def _load_and_preprocess_screenshot(
        self,
        path: Path,
    ) -> MatLike:
        """Load an image from disk. Raise FileNotFoundError if missing."""
        img = cv2.imread(str(path))
        if img is None:
            msg = f"Failed to load screenshot at {path}"
            raise FileNotFoundError(msg)
        return img

    def _crop_desktop_roi(self, img: MatLike) -> MatLike:
        """Crop the bottom UI chrome (e.g., Windows taskbar) from the screenshot.

        The `60` px value is a heuristic — adjust if your environment has a different taskbar height.
        """
        h, w = img.shape[:2]
        return img[0 : h - TASKBAR_HEIGHT_PX, 0:w]

    def _run_template_passes(
        self,
        roi: MatLike,
        icon: Path | None,
        target_size: int,
        config: dict[str, Any],
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Run enabled template matching passes in parallel and collect hits.

        The design runs independent passes (color, lab, edge, gray, orb) concurrently
        because they are CPU-bound and independent; results are merged later.
        """
        if not icon:
            return []
        self._log(
            f"START: Template matching suite (Size: {target_size}px)",
            callback=callback,
            progress=10,
        )
        all_hits: list[Candidate] = []

        # Default core count used if config not provided.
        num_workers = int(config.get("num_cores", 6))

        # Pass configurations (config key, human name, method)
        template_pass_configs = [
            ("use_color", "Color Pass", self._run_color_pass),
            ("use_lab", "CIELAB Pass", self._run_lab_pass),
            ("use_edge", "Edge Pass", self._run_edge_pass),
            ("use_gray", "Grayscale Pass", self._run_gray_pass),
            ("use_orb", "ORB Pass", self._run_orb_pass),
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    self._timed_wrapper,
                    name,
                    func,
                    roi,
                    icon,
                    target_size if name != "ORB Pass" else 0,
                    lambda *a, **kw: self._log(*a, callback=callback, **kw),
                )
                for config_key, name, func in template_pass_configs
                if config.get(config_key)
            ]

            # Multiscale variants can help when icons are differently scaled.
            if config.get("use_multiscale"):
                futures.extend(
                    [
                        executor.submit(
                            self._timed_wrapper,
                            f"Scale {s_factor}x",
                            self._run_scaled_pass,
                            roi,
                            icon,
                            target_size,
                            s_factor,
                            lambda *a, **kw: self._log(*a, callback=callback, **kw),
                        )
                        for s_factor in MULTISCALE_FACTORS
                    ],
                )

            # Collect results as they complete. Aborts if requested.
            for future in concurrent.futures.as_completed(futures):
                if self.should_abort():
                    break
                all_hits.extend(future.result())

        self._log("Template Suite Completed", callback=callback, progress=40)
        return all_hits

    def _run_ocr(
        self,
        roi: MatLike,
        text_query: str,
        psm: int,
        config: dict[str, Any],
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Run global OCR sweep if requested by config and return OCR-based candidates."""
        if not text_query or not config.get("use_ocr"):
            return []
        t0 = time.time()
        hits = self._ocr_search_deep(
            roi,
            text_query,
            psm,
            config,
            lambda *a, **kw: self._log(*a, callback=callback, **kw),
        )
        self.perf_stats.append(
            PerfStat("OCR Global Sweep", (time.time() - t0) * 1000, len(hits)),
        )
        self._log("OCR Sweep Completed", callback=callback, progress=80)
        return hits

    def _validate_templates(
        self,
        icon: Path | None,
        templates: list[Candidate],
    ) -> list[Candidate]:
        """Optionally validate template hits against the template's aspect ratio."""
        return (
            self._validate_spatial_consistency(icon, templates)
            if icon and templates
            else templates
        )

    def _targeted_recovery(
        self,
        roi: MatLike,
        templates: list[Candidate],
        text_query: str,
        target_size: int,
        config: dict[str, Any],
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Perform local OCR around top template matches to recover labels missed globally."""
        if not templates or not text_query:
            return []
        # Limit the recovery queue size to bound runtime (top-N).
        queue = templates[:RECOVERY_QUEUE_LIMIT]
        self._log(
            f"RECOVERY: Verifying {len(queue)} candidates...",
            callback=callback,
            progress=82,
        )
        t0 = time.time()
        hits = self._targeted_label_recovery(
            roi,
            queue,
            text_query,
            target_size,
            config,
            lambda *a, **kw: self._log(*a, callback=callback, **kw),
        )
        self.perf_stats.append(
            PerfStat("Targeted Recovery", (time.time() - t0) * 1000, len(hits)),
        )
        return hits

    def _report_results(
        self,
        canvas: MatLike,
        candidates: list[Candidate],
        t0: float,
        callback: LogCallback | None = None,
    ) -> None:
        """Draw debug visualization, emit benchmark report, and call the callback with a summary table."""
        final_viz = self._draw_results(canvas, candidates)
        self._gui_benchmark_report((time.time() - t0) * 1000, callback)
        if callback and candidates:
            header = f"| {'ID':<4} | {'Method':<15} | {'Score':<8} | {'Coords':<15} |"
            divider = f"|{'-' * 6}|{'-' * 17}|{'-' * 10}|{'-' * 17}|"
            table = ["### Candidate Detection Summary", header, divider]
            for i, c in enumerate(candidates):
                coords = f"({c.x}, {c.y})"
                table.append(
                    f"| {i + 1:<4} | {c.method:<15} | {c.score:<8.2f} | {coords:<15} |",
                )
            callback("\n".join(table), "INFO", 100)
        self._log(
            f"FINISH: Found {len(candidates)} total candidates",
            final_viz,
            callback=callback,
            progress=100,
        )

    def _targeted_label_recovery(
        self,
        img: MatLike,
        templates: list[Candidate],
        query: str,
        target_size: int,
        config: dict[str, Any],
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Recover missing text labels by performing local OCR around template match hits.

        Notes on windowing math:
           - We expand horizontally (pad) to try to capture text that is left/right of the icon.
           - We extend vertically below the icon (1.6x) because many UIs render labels under icons.
           - All values below are heuristics and should be tuned for non-standard environments.
        """
        recovered: list[Candidate] = []
        q_lower = query.lower()
        for idx, template_hit in enumerate(templates):
            if self.should_abort():
                break

            # Heuristic search window size around a matched icon center.
            pad = int(target_size * RECOVERY_HORIZONTAL_PAD_FACTOR)
            # y1 starts slightly above the icon center; y2 extends well below the icon.
            y1, y2 = (
                template_hit.y
                + (target_size // 2)
                - 5,  # small upward bias to include label overlap above center
                template_hit.y + int(target_size * RECOVERY_VERTICAL_EXTEND_FACTOR),
            )
            x1, x2 = (
                template_hit.x - (target_size // 2) - pad,
                template_hit.x + (target_size // 2) + pad,
            )

            # Clamp ROI coordinates so we don't index out of bounds.
            roi = img[
                max(0, y1) : min(img.shape[0], y2),
                max(0, x1) : min(img.shape[1], x2),
            ]
            if roi.size == 0:
                # rare case: template near image edge
                continue

            # Improve OCR by resizing and binarizing.
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            up = cv2.resize(
                gray,
                (0, 0),
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_LINEAR,
            )
            _, thresh = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if config.get("use_adaptive", False):
                # Optional contrast enhancement for hard-to-read labels.
                thresh = self._enhance_contrast_adaptive(thresh)

            cb(f"RECOVERY: Scanning Label ROI {idx + 1}", thresh, progress=82 + idx)

            try:
                # Quick OCR on the small local ROI. Use a single-line PSM for label snippets.
                data = pytesseract.image_to_data(
                    thresh,
                    output_type=pytesseract.Output.DICT,
                    config="--oem 3 --psm 7",
                    timeout=2,
                )
                for txt in data.get("text", []):
                    normalized_token = str(txt).strip().lower()
                    if len(normalized_token) < 2:
                        continue
                    similarity_score = SequenceMatcher(
                        None,
                        q_lower,
                        normalized_token,
                    ).ratio()

                    # Adjust score based on exact/partial match
                    if normalized_token == q_lower:
                        score = 0.95
                    elif q_lower in normalized_token:
                        score = 0.6 + 0.35 * similarity_score
                    else:
                        score = similarity_score * 0.5  # optional low confidence

                    # Only accept reasonably high scores
                    if score >= OCR_RECOVERY_THRESHOLD:
                        recovered.append(
                            Candidate(
                                template_hit.x,
                                template_hit.y,
                                score,
                                "roi_recovery",
                            ),
                        )
                        break

            except Exception as e:
                # Continue on OCR errors for the ROI (timeout, engine issues, etc.)
                cb(f"RECOVERY ERROR: Failed to process ROI {idx + 1}: {e}", lvl="ERROR")
                continue
        return recovered

    def _ocr_search_deep(
        self,
        roi_img: MatLike,
        query: str,
        psm: int,
        config: dict[str, Any],
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Perform a multi-pass deep OCR search using various preprocessing modes.

        Modes: a set of pre-processing pipelines tuned to different text appearances.
        For each mode we run Tesseract with an appropriate page segmentation mode (PSM).
        """
        normalized_query = query.strip().lower()
        raw_results: list[Candidate] = []
        modes = [1, 2, 5, 8, 11, 12]
        lang = config.get("ocr_lang", "eng")

        for i, mode_id in enumerate(modes):
            if self.should_abort():
                break
            progress_val = 40 + (i * 6)

            proc = self._apply_deep_ocr_preprocessing(roi_img, mode_id)

            if config.get("use_adaptive", False):
                proc = self._enhance_contrast_adaptive(proc)

            # Ensure current_psm is always defined; some modes override PSM to 7 for best results.
            current_psm = 7 if mode_id == 12 else psm

            cb(f"PROCESS: OCR Pass {mode_id}", proc, progress=progress_val)

            tess_config = f"--oem 3 --psm {current_psm} -c preserve_interword_spaces=1"
            data = pytesseract.image_to_data(
                proc,
                output_type=pytesseract.Output.DICT,
                config=tess_config,
                lang=lang,
            )

            # If the preprocessing scaled the image, we compensate when converting to ROI coords.
            coord_scale_factor = 2.5 if mode_id in [5, 11, 12] else 1.0

            # Tesseract's image_to_data returns a dictionary of parallel arrays.
            # Each index corresponds to one detected text token:
            #   text[i]   -> recognized string
            #   conf[i]   -> confidence score
            #   left[i]   -> bounding box x coordinate
            #   top[i]    -> bounding box y coordinate
            #   width[i]  -> bounding box width
            #   height[i] -> bounding box height
            # All arrays share the same length, so index `i` represents one token.
            texts = data.get("text", [])
            confs = data.get("conf", [])
            lefts = data.get("left", [])
            tops = data.get("top", [])
            widths = data.get("width", [])
            heights = data.get("height", [])

            for j, txt in enumerate(texts):
                tc = str(txt).strip().lower()
                # Defensive: sometimes Tesseract returns non-numeric conf strings; cast carefully.
                try:
                    conf = float(confs[j])
                except Exception:
                    # treat unknown confidence as low
                    conf = 0.0
                if conf < OCR_MIN_CONFIDENCE or len(tc) < OCR_MIN_TOKEN_LENGTH:
                    # Skip extremely low-confidence or very short tokens (likely noise).
                    continue
                sim = SequenceMatcher(None, normalized_query, tc).ratio()
                # Accept either substring match or moderate similarity.
                if tc == normalized_query:
                    score = 1.0  # exact match
                elif normalized_query in tc:
                    score = (
                        0.5 + 0.5 * sim
                    )  # partial match; base 0.5 + similarity bonus
                else:
                    score = sim  # fallback: similarity only

                raw_results.append(
                    Candidate(
                        int((lefts[j] + widths[j] / 2) / coord_scale_factor),
                        int((tops[j] + heights[j] / 2) / coord_scale_factor),
                        score,
                        f"ocr_m{mode_id}",
                    ),
                )
        # Deduplicate overlapping OCR hits (small radius) before returning.
        return self._deduplicate_with_overlap_logic(raw_results)

    def _run_color_pass(
        self,
        roi_img: MatLike,
        p: Path,
        template_width: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run a standard BGR color-based template matching pass."""
        template_img, template_mask, template_height = self._prep_tpl(p, template_width)
        match_result = cv2.matchTemplate(
            roi_img,
            template_img,
            cv2.TM_CCOEFF_NORMED,
            mask=template_mask,
        )
        cb("PREP: Color Template", template_img, progress=15)
        return self._extract_tpl_locs(
            match_result,
            TPL_COLOR_THRESHOLD,
            template_width,
            template_height,
            "tpl_color",
        )

    def _run_lab_pass(
        self,
        roi_img: MatLike,
        p: Path,
        template_width: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run template matching in CIELAB to be more robust to illumination changes."""
        tpl_b, _, template_height = self._prep_tpl(p, template_width)
        match_result = cv2.matchTemplate(
            cv2.cvtColor(roi_img, cv2.COLOR_BGR2Lab),
            cv2.cvtColor(tpl_b, cv2.COLOR_BGR2Lab),
            cv2.TM_CCOEFF_NORMED,
        )
        cb("PREP: Lab Template", tpl_b, progress=20)
        return self._extract_tpl_locs(
            match_result,
            TPL_LAB_THRESHOLD,
            template_width,
            template_height,
            "tpl_lab",
        )

    def _run_edge_pass(
        self,
        roi_img: MatLike,
        p: Path,
        template_width: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run edge-map based matching (Canny) — useful when color is misleading."""
        tpl_b, _, template_height = self._prep_tpl(p, template_width)
        roi_edges = cv2.Canny(
            cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY),
            CANNY_LOW_THRESHOLD,
            CANNY_HIGH_THRESHOLD,
        )
        template_edges = cv2.Canny(
            cv2.cvtColor(tpl_b, cv2.COLOR_BGR2GRAY),
            CANNY_LOW_THRESHOLD,
            CANNY_HIGH_THRESHOLD,
        )
        match_result = cv2.matchTemplate(
            roi_edges,
            template_edges,
            cv2.TM_CCOEFF_NORMED,
        )
        cb("PREP: Edge Map", roi_edges, progress=25)
        # edge matches are lower-confidence -> lower threshold
        return self._extract_tpl_locs(
            match_result,
            TPL_EDGE_THRESHOLD,
            template_width,
            template_height,
            "tpl_edge",
        )

    def _run_gray_pass(
        self,
        roi_img: MatLike,
        p: Path,
        template_width: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run grayscale template matching (intensity-based)."""
        tpl_b, _, template_height = self._prep_tpl(p, template_width)

        gray_r = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray_t = cv2.cvtColor(tpl_b, cv2.COLOR_BGR2GRAY)

        match_result = cv2.matchTemplate(
            gray_r,
            gray_t,
            cv2.TM_CCOEFF_NORMED,
        )

        cb("PREP: Grayscale Template", gray_r, progress=23)
        return self._extract_tpl_locs(
            match_result,
            TPL_GRAY_THRESHOLD,
            template_width,
            template_height,
            "tpl_gray",
        )

    def _run_orb_pass(
        self,
        roi_img: MatLike,
        p: Path,
        _template_width: int,
        _cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run ORB feature matching to find clusters of keypoint matches.

        ORB is helpful for icons that have distinctive keypoints but poor template correlation.
        We return a single cluster center if enough matches exist.
        """
        template_img = cv2.imread(str(p))
        if template_img is None:
            return []
        orb = cv2.ORB.create(nfeatures=1000)
        _template_keypoints, template_descriptors = orb.detectAndCompute(
            template_img,
            None,
        )
        roi_keypoints, roi_descriptors = orb.detectAndCompute(roi_img, None)
        if template_descriptors is None or roi_descriptors is None:
            return []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(
            bf.match(template_descriptors, roi_descriptors),
            key=lambda x: x.distance,
        )
        # Threshold: need a minimum number of good matches to trust the cluster.
        if len(matches) > ORB_MIN_MATCHES:
            matched_points = np.array(
                [roi_keypoints[i.trainIdx].pt for i in matches[:ORB_SAMPLE_POINTS]],
                dtype=np.float32,
            )
            center = np.mean(matched_points, axis=0)
            # ORB center reported as candidate; confidence is high but not fused with OCR.
            return [Candidate(int(center[0]), int(center[1]), 0.9, "orb")]
        return []

    def _run_scaled_pass(
        self,
        roi_img: MatLike,
        p: Path,
        template_width: int,
        scale_factor: float,
        _cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run template matching at a specific scale factor (multiscale matching)."""
        scaled_template_width = int(template_width * scale_factor)
        template_img, template_mask, template_height = self._prep_tpl(
            p,
            scaled_template_width,
        )
        # If template is bigger than ROI skip to avoid errors.
        if (
            template_img.shape[0] > roi_img.shape[0]
            or template_img.shape[1] > roi_img.shape[1]
        ):
            return []
        match_result = cv2.matchTemplate(
            roi_img,
            template_img,
            cv2.TM_CCOEFF_NORMED,
            mask=template_mask,
        )
        return self._extract_tpl_locs(
            match_result,
            0.65,
            scaled_template_width,
            template_height,
            f"scale_{scale_factor}",
        )

    def _prep_tpl(
        self,
        path: Path,
        template_width: int,
    ) -> tuple[MatLike, MatLike | None, int]:
        """Prepare a template image by resizing and extracting alpha masks if available.

        Returns (tpl_rgb, mask_or_none, computed_height)
        """
        template_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if template_raw is None:
            msg = f"Failed to load template at {path}"
            raise FileNotFoundError(msg)

        # Compute scaling to requested template width (tw).
        scale = template_width / template_raw.shape[1]
        # computed template height after scaling
        template_height = int(template_raw.shape[0] * scale)

        # If image has an alpha channel, extract and resize mask to allow masked matching.
        mask = (
            cv2.resize(template_raw[:, :, 3], (template_width, template_height))
            if template_raw.shape[-1] == 4
            else None
        )
        template_rgb = cv2.resize(
            template_raw[:, :, 0:3] if mask is not None else template_raw,
            (template_width, template_height),
        )
        return template_rgb, mask, template_height

    def _extract_tpl_locs(
        self,
        match_map: MatLike,
        score_threshold: float,
        template_width: int,
        template_height: int,
        method: str,
    ) -> list[Candidate]:
        """Extract candidate coordinates and populate bounding boxes for geometry validation.

        Note: OpenCV.matchTemplate returns top-left coordinates. We convert to center coords
        so candidates are consistently represented by their center.
        """
        match_locations = np.where(match_map >= score_threshold)
        candidates: list[Candidate] = []
        # zip(*locs[::-1]) iterates (x, y) pairs
        for top_left in zip(*match_locations[::-1], strict=False):
            match_score = float(match_map[top_left[1], top_left[0]])
            # Skip invalid scores
            if not np.isfinite(match_score):
                continue

            bbox = (int(top_left[0]), int(top_left[1]), template_width, template_height)
            candidates.append(
                Candidate(
                    x=int(top_left[0] + template_width // 2),
                    y=int(top_left[1] + template_height // 2),
                    score=match_score,
                    method=method,
                    bbox=bbox,
                ),
            )
        return candidates

    def _detect_desktop_icon_size(
        self,
        roi: MatLike,
        config: dict[str, Any] | None = None,
    ) -> int:
        """Detect the dominant desktop icon width using contour widths.

        Configurable min/max icon width via:
          - config["min_icon_width"] (default 30)
          - config["max_icon_width"] (default 150)
        """
        cfg = config or {}
        min_w = cfg.get("min_icon_width", ICON_MIN_WIDTH)
        max_w = cfg.get("max_icon_width", ICON_MAX_WIDTH)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        candidate_widths = [
            cv2.boundingRect(c)[2]
            for c in contours
            if min_w < cv2.boundingRect(c)[2] < max_w
        ]

        # Fallback to a reasonable default when detection fails.
        if not candidate_widths:
            return DEFAULT_ICON_SIZE

        # Choose the most frequent width (mode)
        width_counts: dict[int, Any] = {}
        for w in candidate_widths:
            width_counts[w] = width_counts.get(w, 0) + 1

        dominant_width = max(width_counts.keys(), key=lambda w: width_counts[w])
        return int(dominant_width)

    def _draw_results(self, canvas: MatLike, candidates: list[Candidate]) -> MatLike:
        """Draw detection markers and labels onto the debug canvas for visualization."""
        viz = canvas.copy()
        for i, c in enumerate(candidates):
            # Color scheme: fused = green, ocr-related = yellow, others = cyan
            color = (
                (0, 255, 0)
                if c.method == "fused_match"
                else (0, 255, 255)
                if "ocr" in c.method
                else (255, 255, 0)
            )
            cv2.drawMarker(
                viz,
                (c.x, c.y),
                color,
                cv2.MARKER_SQUARE
                if c.method == "fused_match"
                else cv2.MARKER_TILTED_CROSS,
                25,
                2,
            )
            cv2.putText(
                viz,
                f"#{i + 1} {c.method}",
                (c.x + 18, c.y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        return viz

    def _non_max_suppression(
        self,
        candidates: list[Candidate],
        template_size: int,
    ) -> list[Candidate]:
        """Apply non-maximum suppression to remove overlapping template hits.

        Implementation notes:
          - Keep top-N (200) by score to bound cost of pairwise checking.
          - Compare squared distances to avoid sqrt until necessary.
        """
        if not candidates:
            return []
        sorted_hits = sorted(candidates, key=lambda x: x.score, reverse=True)[
            :MAX_TEMPLATE_HITS
        ]
        kept_candidates: list[Candidate] = []
        # Distance threshold squared: 60% of template height by default
        distance_threshold_sq = (template_size * NMS_RADIUS_FACTOR) ** 2
        for c in sorted_hits:
            if not any(
                ((c.x - a.x) ** 2 + (c.y - a.y) ** 2) < distance_threshold_sq
                for a in kept_candidates
            ):
                kept_candidates.append(c)
        return kept_candidates

    def _timed_wrapper(
        self,
        name: str,
        func: Callable[P, list[Candidate]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> list[Candidate]:
        """Wrap a function call to measure its execution time and record a PerfStat."""
        t0 = time.time()
        results = func(*args, **kwargs)
        self.perf_stats.append(PerfStat(name, (time.time() - t0) * 1000, len(results)))
        return results

    def _gui_benchmark_report(
        self,
        total_time: float,
        callback: LogCallback | None,
    ) -> None:
        """Generate and send a formatted benchmark report via the provided callback."""
        if not callback:
            return
        report = [f"VISION ENGINE BENCHMARK ({total_time:.0f}ms)"]
        new_entries = [
            f"| {stat.name:<20} | {stat.duration_ms:<8.0f} | {stat.items_found:<8} |"
            for stat in self.perf_stats
        ]
        report.extend(new_entries)
        callback("\n".join(report), "HEAD", 100)

    def _apply_deep_ocr_preprocessing(self, roi_img: MatLike, mode_id: int) -> MatLike:
        """Apply a specific image preprocessing pipeline for deep OCR detection.

        Each pipeline is tuned to a scenario (high-contrast, inverted text, upscaling, morphological top-hat).
        Choose mode numbers carefully — they map to preprocessing strategies used in _ocr_search_deep.
        """
        gray_image = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        if mode_id == 1:
            # Standard Otsu binarization
            return cv2.threshold(
                gray_image,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]
        if mode_id == 2:
            # Inverted + Otsu — helpful for light-on-dark text.
            return cv2.threshold(
                cv2.bitwise_not(gray_image),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]
        if mode_id == 5:
            # Upscale to improve OCR on small fonts.
            return cv2.resize(
                gray_image,
                (0, 0),
                fx=2.5,
                fy=2.5,
                interpolation=cv2.INTER_CUBIC,
            )
        if mode_id in [11, 12]:
            # Upscale + top-hat morphological op to highlight strokes.
            upscaled_image = cv2.resize(
                gray_image,
                (0, 0),
                fx=2.5,
                fy=2.5,
                interpolation=cv2.INTER_CUBIC,
            )
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            return cv2.threshold(
                cv2.morphologyEx(upscaled_image, cv2.MORPH_TOPHAT, morph_kernel),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]
        if mode_id == 8:
            # CLAHE + Otsu: good for uneven lighting.
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
            return cv2.threshold(
                clahe.apply(gray_image),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]
        return gray_image

    def _deduplicate_with_overlap_logic(
        self,
        raw_results: list[Candidate],
    ) -> list[Candidate]:
        """Deduplicate candidate results by filtering out overlapping hits within a radius.

        We sort by score desc and keep the highest scoring candidate in local neighborhoods.
        """
        filtered_candidates: list[Candidate] = []
        for candidate in sorted(raw_results, key=lambda x: x.score, reverse=True):
            # Radius of TOKEN_DEDUP_RADIUS_PX chosen as small token-level dedupe for OCR results.
            if not any(
                np.sqrt(
                    (candidate.x - kept_candidate.x) ** 2
                    + (candidate.y - kept_candidate.y) ** 2,
                )
                < TOKEN_DEDUP_RADIUS_PX
                for kept_candidate in filtered_candidates
            ):
                filtered_candidates.append(candidate)
        return filtered_candidates

    @staticmethod
    def _euclidean_distance(a: Candidate, b: Candidate) -> float:
        """Return Euclidean distance between two candidate centers."""
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def _finalize_results(
        self,
        template_hits: list[Candidate],
        ocr_hits: list[Candidate],
        template_size: int,
        score_threshold: float,
    ) -> list[Candidate]:
        """Fuse template and OCR matches and apply final filtering and NMS.

        Fusion:
          - If a template hit and OCR hit are within 1.5 * ts (ts = template size),
            consider them the same object and create a fused candidate with raised score.
        Post-processing:
          - Preserve unmatched template-only and OCR-only hits.
          - Apply a final proximity-based dedupe using ts * 0.7 and threshold filter.
        """
        final_list: list[Candidate] = []
        matched_ocr_indices, matched_template_indices = set[int](), set[int]()
        for template_index, template_hit in enumerate(template_hits):
            for ocr_index, ocr_hit in enumerate(ocr_hits):
                if ocr_index in matched_ocr_indices:
                    continue
                # Spatial fusion threshold (FUSION_DISTANCE_FACTOR * template size) — heuristic.
                if (
                    self._euclidean_distance(template_hit, ocr_hit)
                    < template_size * FUSION_DISTANCE_FACTOR
                ):
                    matched_ocr_indices.add(ocr_index)
                    matched_template_indices.add(template_index)
                    final_list.append(
                        Candidate(
                            template_hit.x,
                            template_hit.y,
                            max(template_hit.score, ocr_hit.score) + 0.1,
                            "fused_match",
                            template_hit.score,
                            ocr_hit.score,
                        ),
                    )
                    break
        # Add leftovers (template-only and OCR-only)
        final_list.extend(
            [
                t
                for i, t in enumerate(template_hits)
                if i not in matched_template_indices
            ],
        )
        final_list.extend(
            [o for j, o in enumerate(ocr_hits) if j not in matched_ocr_indices],
        )

        # Final proximity dedupe: keep highest scoring per local neighborhood.
        deduped_results: list[Candidate] = []
        for c in sorted(final_list, key=lambda x: x.score, reverse=True):
            if not any(
                np.sqrt(
                    (c.x - existing_candidate.x) ** 2
                    + (c.y - existing_candidate.y) ** 2,
                )
                < template_size * FINAL_DEDUP_RADIUS_FACTOR
                for existing_candidate in deduped_results
            ):
                deduped_results.append(c)

        # Final threshold filter (score must be >= thr).
        return [
            existing_candidate
            for existing_candidate in deduped_results
            if existing_candidate.score >= score_threshold
        ]

    def _validate_spatial_consistency(
        self,
        icon_path: Path,
        hits: list[Candidate],
    ) -> list[Candidate]:
        """Filter candidates that deviate significantly from the template's aspect ratio.

        Adjust each hit's geometry_score based on aspect ratio deviation and slightly
        penalize the visual score if the shape is inconsistent.
        """
        template_img = cv2.imread(str(icon_path))
        if template_img is None or not hits:
            return hits

        template_height, template_width = template_img.shape[:2]
        target_ratio = template_width / template_height

        validated_hits: list[Candidate] = []
        for hit in hits:
            if hit.bbox:
                _, _, w_hit, h_hit = hit.bbox
                current_ratio = w_hit / h_hit

                # deviation ∈ (0, 1], 1 -> perfect match
                deviation = min(target_ratio, current_ratio) / max(
                    target_ratio,
                    current_ratio,
                )
                hit.geometry_score = float(deviation)

                # Slightly adjust hit score by geometry agreement (0.8 -> 1.0 factor).
                # Keep 80% of original score minimum; reward up to +20% for perfect ratio match
                hit.score *= 0.8 + (0.2 * deviation)

            validated_hits.append(hit)

        return validated_hits

    def _enhance_contrast_adaptive(self, img: MatLike) -> MatLike:
        """Apply adaptive histogram equalization (CLAHE) to improve text visibility.

        Works for grayscale images and color images (applies CLAHE on L channel in LAB).
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
