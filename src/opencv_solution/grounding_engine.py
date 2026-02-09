"""DesktopGroundingEngine.

Provides computer vision and OCR capabilities for locating desktop UI elements.

Summary / Quickstart
--------------------
- Call DesktopGroundingEngine.locate_elements(screenshot_path, icon_image, text_query, ...)
  to run template-matching and OCR passes and get a ranked list of Candidate objects.
- Candidate: (x, y, score, method, img_score, txt_score, bbox, geometry_score)
- Engine uses multiple independent strategies in parallel:
    * Template matching (color, CIELAB, grayscale, edge)
    * ORB feature matching (keypoint clustering)
    * Multi-pass OCR (Tesseract) with local recovery around template hits
- Configuration flags (pass booleans, cores, multiscale, OCR language/options) influence which passes run.

Design intent comments appear inline near heuristics, thresholds, and coordinate math.
"""

import logging
import platform

if platform.system() == "Windows":
    try:
        import ctypes

        # Ensure the process is DPI aware so screenshots and coordinates align on Windows.
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception("DPI awareness failed")

import concurrent.futures
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, ParamSpec

import cv2
import numpy as np
import pytesseract

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cv2.typing import MatLike
    from numpy.typing import NDArray


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
        self.debug_frame: NDArray | None = None

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
        """Select the highest-ranking candidate based on the specified priority strategy.

        Strategies:
          - 'fusion': prefer fused template+OCR matches (strong visual + text agreement)
          - 'text': prefer OCR-derived matches
          - fallback: highest raw score across methods
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
        scale: float = 2.0,
        config: dict[str, Any] | None = None,
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Locate screen elements by orchestrating template matching and OCR sweeps.

        High-level flow:
          1. Load screenshot, crop UI chrome (taskbar) region
          2. Detect approximate desktop icon size
          3. Run chosen template passes in parallel
          4. Run global OCR sweep
          5. Validate templates, run targeted OCR recovery around template hits
          6. Fuse template + OCR hits, apply filtering & NMS, and return final candidates
        """
        self.perf_stats = []
        safe_config: dict[str, Any] = config or {}
        t0 = time.time()

        img = self._load_and_preprocess_screenshot(screenshot_path)
        full_img = self._load_and_preprocess_screenshot(screenshot_path)
        desktop_roi = self._crop_desktop_roi(img)
        self._log("INIT: Screenshot Loaded", desktop_roi, callback, progress=5)

        if self.should_abort():
            return []

        # Heuristic: detect icon size to scale template appropriately.
        target_size = self._detect_desktop_icon_size(
            desktop_roi,
            lambda *a, **kw: self._log(*a, callback=callback, **kw),
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
            scale,
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
            safe_config,
        )

        self._report_results(full_img, final_candidates, t0, callback)
        return final_candidates

    def _log(
        self,
        msg: str,
        frame: NDArray | None = None,
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

    def _crop_desktop_roi(self, img: NDArray) -> NDArray:
        """Crop the bottom UI chrome (e.g., Windows taskbar) from the screenshot.

        The `60` px value is a heuristic — adjust if your environment has a different taskbar height.
        """
        h, w = img.shape[:2]
        return img[0 : h - 60, 0:w]

    def _run_template_passes(
        self,
        roi: NDArray,
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
        pass_configs = [
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
                for cfg_key, name, func in pass_configs
                if config.get(cfg_key)
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
                        for s_factor in [0.8, 1.25]
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
        roi: NDArray,
        text_query: str,
        psm: int,
        scale: float,
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
            scale,
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
        roi: NDArray,
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
        queue = templates[:12]
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
        canvas: NDArray,
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
        img: NDArray,
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
        for idx, t in enumerate(templates):
            if self.should_abort():
                break

            # Heuristic search window size around a matched icon center.
            pad = int(target_size * 0.4)
            # y1 starts slightly above the icon center; y2 extends well below the icon.
            y1, y2 = t.y + (target_size // 2) - 5, t.y + int(target_size * 1.6)
            x1, x2 = t.x - (target_size // 2) - pad, t.x + (target_size // 2) + pad

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
                data: dict[str, list[Any]] = pytesseract.image_to_data(
                    thresh,
                    output_type=pytesseract.Output.DICT,
                    config="--oem 3 --psm 7",
                    timeout=2,
                )
                for txt in data.get("text", []):
                    tc = str(txt).strip().lower()
                    if len(tc) < 2:
                        continue
                    sim = SequenceMatcher(None, q_lower, tc).ratio()

                    # Adjust score based on exact/partial match
                    if tc == q_lower:
                        score = 0.95
                    elif q_lower in tc:
                        score = 0.6 + 0.35 * sim
                    else:
                        score = sim * 0.5  # optional low confidence

                    # Only accept reasonably high scores
                    if score >= 0.5:
                        recovered.append(Candidate(t.x, t.y, score, "roi_recovery"))
                        break

            except Exception as e:
                # Continue on OCR errors for the ROI (timeout, engine issues, etc.)
                cb(f"RECOVERY ERROR: Failed to process ROI {idx + 1}: {e}", lvl="ERROR")
                continue
        return recovered

    def _ocr_search_deep(
        self,
        r: MatLike,
        q: str,
        psm: int,
        _s: float,
        _config: dict[str, Any],
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Perform a multi-pass deep OCR search using various preprocessing modes.

        Modes: a set of pre-processing pipelines tuned to different text appearances.
        For each mode we run Tesseract with an appropriate page segmentation mode (PSM).
        """
        qc, raw_results = q.strip().lower(), []
        modes = [1, 2, 5, 8, 11, 12]
        lang = _config.get("ocr_lang", "eng")

        for i, p_num in enumerate(modes):
            if self.should_abort():
                break
            progress_val = 40 + (i * 6)

            proc = self._apply_deep_ocr_preprocessing(r, p_num)

            if _config.get("use_adaptive", False):
                proc = self._enhance_contrast_adaptive(proc)

            # Ensure current_psm is always defined; some modes override PSM to 7 for best results.
            current_psm = 7 if p_num == 12 else psm

            cb(f"PROCESS: OCR Pass {p_num}", proc, progress=progress_val)

            tess_config = f"--oem 3 --psm {current_psm} -c preserve_interword_spaces=1"
            data: dict[str, list[Any]] = pytesseract.image_to_data(
                proc,
                output_type=pytesseract.Output.DICT,
                config=tess_config,
                lang=lang,
            )

            # If the preprocessing scaled the image, we compensate when converting to ROI coords.
            current_sc = 2.5 if p_num in [5, 11, 12] else 1.0

            texts, confs = data.get("text", []), data.get("conf", [])
            lefts, tops = data.get("left", []), data.get("top", [])
            widths, heights = data.get("width", []), data.get("height", [])

            for j, txt in enumerate(texts):
                tc = str(txt).strip().lower()
                # Defensive: sometimes Tesseract returns non-numeric conf strings; cast carefully.
                try:
                    conf = float(confs[j])
                except Exception:
                    # treat unknown confidence as low
                    conf = 0.0
                if conf < 10 or len(tc) < 2:
                    # Skip extremely low-confidence or very short tokens (likely noise).
                    continue
                sim = SequenceMatcher(None, qc, tc).ratio()
                # Accept either substring match or moderate similarity.
                if tc == qc:
                    score = 1.0  # exact match
                elif qc in tc:
                    score = (
                        0.5 + 0.5 * sim
                    )  # partial match; base 0.5 + similarity bonus
                else:
                    score = sim  # fallback: similarity only

                raw_results.append(
                    Candidate(
                        int((lefts[j] + widths[j] / 2) / current_sc),
                        int((tops[j] + heights[j] / 2) / current_sc),
                        score,
                        f"ocr_m{p_num}",
                    ),
                )
        # Deduplicate overlapping OCR hits (small radius) before returning.
        return self._deduplicate_with_overlap_logic(raw_results)

    def _run_color_pass(
        self,
        r: MatLike,
        p: Path,
        tw: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run a standard BGR color-based template matching pass."""
        tpl, m, th = self._prep_tpl(p, tw)
        res = cv2.matchTemplate(
            r,
            tpl,
            cv2.TM_CCOEFF_NORMED,
            mask=m,
        )
        cb("PREP: Color Template", tpl, progress=15)
        return self._extract_tpl_locs(res, 0.7, tw, th, "tpl_color")

    def _run_lab_pass(
        self,
        r: MatLike,
        p: Path,
        tw: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run template matching in CIELAB to be more robust to illumination changes."""
        tpl_b, _, th = self._prep_tpl(p, tw)
        res = cv2.matchTemplate(
            cv2.cvtColor(r, cv2.COLOR_BGR2Lab),
            cv2.cvtColor(tpl_b, cv2.COLOR_BGR2Lab),
            cv2.TM_CCOEFF_NORMED,
        )
        cb("PREP: Lab Template", tpl_b, progress=20)
        return self._extract_tpl_locs(res, 0.7, tw, th, "tpl_lab")

    def _run_edge_pass(
        self,
        r: MatLike,
        p: Path,
        tw: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run edge-map based matching (Canny) — useful when color is misleading."""
        tpl_b, _, th = self._prep_tpl(p, tw)
        re = cv2.Canny(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), 50, 150)
        te = cv2.Canny(cv2.cvtColor(tpl_b, cv2.COLOR_BGR2GRAY), 50, 150)
        res = cv2.matchTemplate(re, te, cv2.TM_CCOEFF_NORMED)
        cb("PREP: Edge Map", re, progress=25)
        # edge matches are lower-confidence -> lower threshold
        return self._extract_tpl_locs(res, 0.4, tw, th, "tpl_edge")

    def _run_gray_pass(
        self,
        r: MatLike,
        p: Path,
        tw: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run grayscale template matching (intensity-based)."""
        tpl_b, _, th = self._prep_tpl(p, tw)

        gray_r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        gray_t = cv2.cvtColor(tpl_b, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(
            gray_r,
            gray_t,
            cv2.TM_CCOEFF_NORMED,
        )

        cb("PREP: Grayscale Template", gray_r, progress=23)
        return self._extract_tpl_locs(res, 0.65, tw, th, "tpl_gray")

    def _run_orb_pass(
        self,
        r: MatLike,
        p: Path,
        _: int,
        _cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run ORB feature matching to find clusters of keypoint matches.

        ORB is helpful for icons that have distinctive keypoints but poor template correlation.
        We return a single cluster center if enough matches exist.
        """
        tpl = cv2.imread(str(p))
        if tpl is None:
            return []
        orb = cv2.ORB.create(nfeatures=1000)
        _k1, d1 = orb.detectAndCompute(tpl, None)
        k2, d2 = orb.detectAndCompute(r, None)
        if d1 is None or d2 is None:
            return []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        m = sorted(bf.match(d1, d2), key=lambda x: x.distance)
        # Threshold: need a minimum number of good matches to trust the cluster.
        if len(m) > 15:
            pts = np.array([k2[i.trainIdx].pt for i in m[:20]], dtype=np.float32)
            center = np.mean(pts, axis=0)
            # ORB center reported as candidate; confidence is high but not fused with OCR.
            return [Candidate(int(center[0]), int(center[1]), 0.9, "orb")]
        return []

    def _run_scaled_pass(
        self,
        r: MatLike,
        p: Path,
        tw: int,
        sf: float,
        _cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run template matching at a specific scale factor (multiscale matching)."""
        stw = int(tw * sf)
        tpl, m, th = self._prep_tpl(p, stw)
        # If template is bigger than ROI skip to avoid errors.
        if tpl.shape[0] > r.shape[0] or tpl.shape[1] > r.shape[1]:
            return []
        res = cv2.matchTemplate(
            r,
            tpl,
            cv2.TM_CCOEFF_NORMED,
            mask=m,
        )
        return self._extract_tpl_locs(res, 0.65, stw, th, f"scale_{sf}")

    def _prep_tpl(self, path: Path, tw: int) -> tuple[MatLike, MatLike | None, int]:
        """Prepare a template image by resizing and extracting alpha masks if available.

        Returns (tpl_rgb, mask_or_none, computed_height)
        """
        tpl_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if tpl_raw is None:
            msg = f"Failed to load template at {path}"
            raise FileNotFoundError(msg)

        # Compute scaling to requested template width (tw).
        scale = tw / tpl_raw.shape[1]
        # computed template height after scaling
        th = int(tpl_raw.shape[0] * scale)

        # If image has an alpha channel, extract and resize mask to allow masked matching.
        mask = (
            cv2.resize(tpl_raw[:, :, 3], (tw, th)) if tpl_raw.shape[-1] == 4 else None
        )
        tpl = cv2.resize(tpl_raw[:, :, 0:3] if mask is not None else tpl_raw, (tw, th))
        return tpl, mask, th

    def _extract_tpl_locs(
        self,
        res: NDArray,
        thr: float,
        tw: int,
        th: int,
        method: str,
    ) -> list[Candidate]:
        """Extract candidate coordinates and populate bounding boxes for geometry validation.

        Note: OpenCV.matchTemplate returns top-left coordinates. We convert to center coords
        so candidates are consistently represented by their center.
        """
        locs = np.where(res >= thr)
        candidates = []
        # zip(*locs[::-1]) iterates (x, y) pairs
        for pt in zip(*locs[::-1], strict=False):
            score_val = float(res[pt[1], pt[0]])
            # Skip invalid scores
            if not np.isfinite(score_val):
                continue

            bbox = (int(pt[0]), int(pt[1]), tw, th)
            candidates.append(
                Candidate(
                    x=int(pt[0] + tw // 2),
                    y=int(pt[1] + th // 2),
                    score=score_val,
                    method=method,
                    bbox=bbox,
                ),
            )
        return candidates

    def _detect_desktop_icon_size(
        self,
        roi: NDArray,
        _cb: Callable[..., Any],
        config: dict[str, Any] | None = None,
    ) -> int:
        """Detect the dominant desktop icon width using contour widths.

        Configurable min/max icon width via:
          - config["min_icon_width"] (default 30)
          - config["max_icon_width"] (default 150)
        """
        cfg = config or {}
        min_w = cfg.get("min_icon_width", 30)
        max_w = cfg.get("max_icon_width", 150)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sizes = [
            cv2.boundingRect(c)[2]
            for c in cnts
            if min_w < cv2.boundingRect(c)[2] < max_w
        ]

        # Fallback to a reasonable default when detection fails.
        return int(max(set(sizes), key=sizes.count)) if sizes else 64

    def _draw_results(self, canvas: NDArray, candidates: list[Candidate]) -> NDArray:
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
        hits: list[Candidate],
        thresh: int,
    ) -> list[Candidate]:
        """Apply non-maximum suppression to remove overlapping template hits.

        Implementation notes:
          - Keep top-N (200) by score to bound cost of pairwise checking.
          - Compare squared distances to avoid sqrt until necessary.
        """
        if not hits:
            return []
        sh = sorted(hits, key=lambda x: x.score, reverse=True)[:200]
        final: list[Candidate] = []
        # Distance threshold squared: 60% of template height by default
        d_thresh_sq = (thresh * 0.6) ** 2
        for c in sh:
            if not any(
                ((c.x - a.x) ** 2 + (c.y - a.y) ** 2) < d_thresh_sq for a in final
            ):
                final.append(c)
        return final

    def _timed_wrapper(
        self,
        name: str,
        func: Callable[P, list[Candidate]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> list[Candidate]:
        """Wrap a function call to measure its execution time and record a PerfStat."""
        t0 = time.time()
        res = func(*args, **kwargs)
        self.perf_stats.append(PerfStat(name, (time.time() - t0) * 1000, len(res)))
        return res

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

    def _apply_deep_ocr_preprocessing(self, r: MatLike, p_num: int) -> MatLike:
        """Apply a specific image preprocessing pipeline for deep OCR detection.

        Each pipeline is tuned to a scenario (high-contrast, inverted text, upscaling, morphological top-hat).
        Choose mode numbers carefully — they map to preprocessing strategies used in _ocr_search_deep.
        """
        g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        if p_num == 1:
            # Standard Otsu binarization
            return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if p_num == 2:
            # Inverted + Otsu — helpful for light-on-dark text.
            return cv2.threshold(
                cv2.bitwise_not(g),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]
        if p_num == 5:
            # Upscale to improve OCR on small fonts.
            return cv2.resize(g, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        if p_num in [11, 12]:
            # Upscale + top-hat morphological op to highlight strokes.
            up = cv2.resize(g, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            return cv2.threshold(
                cv2.morphologyEx(up, cv2.MORPH_TOPHAT, k),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]
        if p_num == 8:
            # CLAHE + Otsu: good for uneven lighting.
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
            return cv2.threshold(
                clahe.apply(g),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]
        return g

    def _deduplicate_with_overlap_logic(
        self,
        raw_results: list[Candidate],
    ) -> list[Candidate]:
        """Deduplicate candidate results by filtering out overlapping hits within a radius.

        We sort by score desc and keep the highest scoring candidate in local neighborhoods.
        """
        clean: list[Candidate] = []
        for cand in sorted(raw_results, key=lambda x: x.score, reverse=True):
            # Radius of 15 px chosen as small token-level dedupe for OCR results.
            if not any(
                np.sqrt((cand.x - k.x) ** 2 + (cand.y - k.y) ** 2) < 15 for k in clean
            ):
                clean.append(cand)
        return clean

    def _finalize_results(
        self,
        t_hits: list[Candidate],
        o_hits: list[Candidate],
        ts: int,
        thr: float,
        _cfg: dict[str, Any],
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
        used_ocr, used_tpl = set(), set()
        for i, t in enumerate(t_hits):
            for j, o in enumerate(o_hits):
                if j in used_ocr:
                    continue
                # Spatial fusion threshold (1.5 * template size) — heuristic.
                if np.sqrt((t.x - o.x) ** 2 + (t.y - o.y) ** 2) < ts * 1.5:
                    used_ocr.add(j)
                    used_tpl.add(i)
                    final_list.append(
                        Candidate(
                            t.x,
                            t.y,
                            max(t.score, o.score) + 0.1,
                            "fused_match",
                            t.score,
                            o.score,
                        ),
                    )
                    break
        # Add leftovers (template-only and OCR-only)
        final_list.extend([t for i, t in enumerate(t_hits) if i not in used_tpl])
        final_list.extend([o for j, o in enumerate(o_hits) if j not in used_ocr])

        # Final proximity dedupe: keep highest scoring per local neighborhood.
        res: list[Candidate] = []
        for c in sorted(final_list, key=lambda x: x.score, reverse=True):
            if not any(
                np.sqrt((c.x - r.x) ** 2 + (c.y - r.y) ** 2) < ts * 0.7 for r in res
            ):
                res.append(c)

        # Final threshold filter (score must be >= thr).
        return [r for r in res if r.score >= thr]

    def _validate_spatial_consistency(
        self,
        icon_path: Path,
        hits: list[Candidate],
    ) -> list[Candidate]:
        """Filter candidates that deviate significantly from the template's aspect ratio.

        Adjust each hit's geometry_score based on aspect ratio deviation and slightly
        penalize the visual score if the shape is inconsistent.
        """
        tpl = cv2.imread(str(icon_path))
        if tpl is None or not hits:
            return hits

        h_tpl, w_tpl = tpl.shape[:2]
        target_ratio = w_tpl / h_tpl

        validated_hits = []
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
