"""Provide computer vision and OCR capabilities for locating desktop elements.

This module contains the DesktopGroundingEngine, which combines template matching,
feature detection, and OCR to identify UI elements and icons on a screen.
"""

import concurrent.futures
import time
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytesseract
from numpy.typing import NDArray


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
    geometry_score: float = 1.0  # Added: spatial consistency score


@dataclass
class PerfStat:
    """Store performance metrics for a specific detection pass."""

    name: str
    duration_ms: float
    items_found: int


class DesktopGroundingEngine:
    """Execute multi-modal searches for UI elements using OpenCV and Tesseract.

    Coordinate various detection strategies including template matching (color,
    CIELAB, edges), ORB feature matching, and deep OCR sweeps.
    """

    def __init__(self, tesseract_path: str) -> None:
        """Initialize the engine with the provided Tesseract executable path."""
        self.tesseract_path: str = tesseract_path
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        self.debug_frame: NDArray | None = None
        self.perf_stats: list[PerfStat] = []
        self.should_abort: Callable[[], bool] = lambda: False

    def select_best_candidate(
        self,
        candidates: list[Candidate],
        priority: str = "fusion",
    ) -> Candidate | None:
        """Select the highest-ranking candidate based on the specified priority strategy."""
        if not candidates:
            return None

        # 1. Priority 'fusion': Look for the specific engine tag 'fused_match'
        if priority == "fusion":
            fused = [c for c in candidates if c.method == "fused_match"]
            if fused:
                return max(fused, key=lambda c: c.score)

        # 2. Text Priority: Focus on OCR results
        if priority == "text":
            ocr_cands = [c for c in candidates if "ocr" in c.method]
            if ocr_cands:
                return max(ocr_cands, key=lambda c: c.score)

        # Fallback: Return the highest absolute score regardless of method
        return max(candidates, key=lambda c: c.score)

    def locate_elements(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        screenshot_path: Path,
        icon_image: Path | None,
        text_query: str,
        threshold: float = 0.5,
        psm: int = 11,
        scale: float = 2.0,
        config: dict[str, Any] | None = None,
        callback: Callable[[str, str, int | None], None] | None = None,
    ) -> list[Candidate]:
        """Locate screen elements by orchestrating template matching and OCR sweeps."""
        self.perf_stats = []
        safe_config: dict[str, Any] = config or {}

        def log_and_show(
            msg: str,
            frame: NDArray | None = None,
            lvl: str = "INFO",
            progress: int | None = None,
        ) -> None:
            if self.should_abort():
                return
            if frame is not None:
                if len(frame.shape) == 2:
                    self.debug_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    self.debug_frame = frame.copy()
            if callback:
                callback(msg, lvl, progress)

        img: NDArray = cv2.imread(str(screenshot_path))

        if safe_config.get("use_adaptive"):
            log_and_show(
                "PRE-PROCESS: Applying Adaptive Contrast Enhancement",
                progress=7,
            )
            img = self._enhance_contrast_adaptive(img)

        h_hay, w_hay = img.shape[:2]
        desktop_roi: NDArray = img[0 : h_hay - 60, 0:w_hay]
        log_and_show("INIT: Screenshot Loaded", desktop_roi, progress=5)

        if self.should_abort():
            return []

        target_size: int = self._detect_desktop_icon_size(desktop_roi, log_and_show)
        num_workers: int = int(safe_config.get("num_cores", 6))
        t0: float = time.time()

        all_template_hits: list[Candidate] = []
        if icon_image:
            log_and_show(
                f"START: Template matching suite (Size: {target_size}px)",
                progress=10,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers,
            ) as executor:
                futures: list[concurrent.futures.Future[list[Candidate]]] = []
                pass_configs = [
                    ("use_color", "Color Pass", self._run_color_pass),
                    ("use_lab", "CIELAB Pass", self._run_lab_pass),
                    ("use_edge", "Edge Pass", self._run_edge_pass),
                    ("use_orb", "ORB Pass", self._run_orb_pass),
                ]

                for cfg_key, name, func in pass_configs:
                    if safe_config.get(cfg_key):
                        futures.append(
                            executor.submit(
                                self._timed_wrapper,
                                name,
                                func,
                                desktop_roi,
                                icon_image,
                                target_size if name != "ORB Pass" else 0,
                                log_and_show,
                            ),
                        )

                if safe_config.get("use_multiscale"):
                    new_futures = [
                        executor.submit(
                            self._timed_wrapper,
                            f"Scale {s_factor}x",
                            self._run_scaled_pass,
                            desktop_roi,
                            icon_image,
                            target_size,
                            s_factor,
                            log_and_show,
                        )
                        for s_factor in [0.8, 1.25]
                    ]
                    futures.extend(new_futures)

                for future in concurrent.futures.as_completed(futures):
                    if self.should_abort():
                        break
                    all_template_hits.extend(future.result())

        log_and_show("Template Suite Completed", progress=40)
        if self.should_abort():
            return []

        ocr_list: list[Candidate] = []
        if text_query and safe_config.get("use_ocr"):
            t_ocr = time.time()
            ocr_list = self._ocr_search_deep(
                desktop_roi,
                text_query,
                psm,
                scale,
                safe_config,
                log_and_show,
            )
            self.perf_stats.append(
                PerfStat(
                    "OCR Global Sweep",
                    (time.time() - t_ocr) * 1000,
                    len(ocr_list),
                ),
            )

        log_and_show("OCR Sweep Completed", progress=80)
        if self.should_abort():
            return []

        templates = self._non_max_suppression(all_template_hits, target_size)

        # New Feature: Verify geometric consistency if an icon was provided
        if icon_image and templates:
            templates = self._validate_spatial_consistency(icon_image, templates)

        if templates and text_query:
            recovery_queue = templates[:12]
            log_and_show(
                f"RECOVERY: Verifying {len(recovery_queue)} candidates...",
                progress=82,
            )
            t_rec = time.time()
            recovery_hits = self._targeted_label_recovery(
                desktop_roi,
                recovery_queue,
                text_query,
                target_size,
                log_and_show,
            )
            ocr_list.extend(recovery_hits)
            self.perf_stats.append(
                PerfStat(
                    "Targeted Recovery",
                    (time.time() - t_rec) * 1000,
                    len(recovery_hits),
                ),
            )

        log_and_show("Finalizing Fusion...", progress=90)
        final_candidates = self._finalize_results(
            templates,
            ocr_list,
            target_size,
            threshold,
            safe_config,
        )

        final_viz = self._draw_results(desktop_roi, final_candidates)
        self._gui_benchmark_report((time.time() - t0) * 1000, callback)

        if callback and final_candidates:
            header = f"| {'ID':<4} | {'Method':<15} | {'Score':<8} | {'Coords':<15} |"
            divider = f"|{'-' * 6}|{'-' * 17}|{'-' * 10}|{'-' * 17}|"
            table = ["### Candidate Detection Summary", header, divider]
            for i, c in enumerate(final_candidates):
                coords = f"({c.x}, {c.y})"
                table.append(
                    f"| {i + 1:<4} | {c.method:<15} | {c.score:<8.2f} | {coords:<15} |",
                )
            callback("\n".join(table), "INFO", 100)

        log_and_show(
            f"FINISH: Found {len(final_candidates)} total candidates",
            final_viz,
            progress=100,
        )
        return final_candidates

    def _targeted_label_recovery(
        self,
        img: NDArray,
        templates: list[Candidate],
        query: str,
        target_size: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Recover missing text labels by performing local OCR around template match hits."""
        recovered: list[Candidate] = []
        q_lower = query.lower()
        for idx, t in enumerate(templates):
            if self.should_abort():
                break
            pad = int(target_size * 0.4)
            y1, y2 = t.y + (target_size // 2) - 5, t.y + int(target_size * 1.6)
            x1, x2 = t.x - (target_size // 2) - pad, t.x + (target_size // 2) + pad

            roi = img[
                max(0, y1) : min(img.shape[0], y2),
                max(0, x1) : min(img.shape[1], x2),
            ]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            up = cv2.resize(
                gray,
                (0, 0),
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_LINEAR,
            )
            _, thresh = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cb(f"RECOVERY: Scanning Label ROI {idx + 1}", thresh, progress=82 + idx)

            try:
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
                    if q_lower in tc or sim > 0.7:
                        recovered.append(Candidate(t.x, t.y, 0.95, "roi_recovery"))
                        break
            except Exception as e:
                cb(f"RECOVERY ERROR: Failed to process ROI {idx + 1}: {e}", lvl="ERROR")
                continue
        return recovered

    def _ocr_search_deep(
        self,
        r: NDArray,
        q: str,
        psm: int,
        _s: float,
        _config: dict[str, Any],
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Perform a multi-pass deep OCR search using various preprocessing modes."""
        qc, raw_results = q.strip().lower(), []
        modes = [1, 2, 5, 8, 11, 12]
        # New Feature: Support for dynamic language settings
        lang = _config.get("ocr_lang", "eng")

        for i, p_num in enumerate(modes):
            if self.should_abort():
                break
            progress_val = 40 + (i * 6)
            proc = self._apply_deep_ocr_preprocessing(r, p_num)
            cb(f"PROCESS: OCR Pass {p_num}", proc, progress=progress_val)

            current_psm = 7 if p_num == 12 else psm
            tess_config = f"--oem 3 --psm {current_psm} -c preserve_interword_spaces=1"
            data: dict[str, list[Any]] = pytesseract.image_to_data(
                proc,
                output_type=pytesseract.Output.DICT,
                config=tess_config,
                lang=lang,
            )
            current_sc = 2.5 if p_num in [5, 11, 12] else 1.0

            texts, confs = data.get("text", []), data.get("conf", [])
            lefts, tops = data.get("left", []), data.get("top", [])
            widths, heights = data.get("width", []), data.get("height", [])

            for j, txt in enumerate(texts):
                tc, conf = str(txt).strip().lower(), float(confs[j])
                if conf < 10 or len(tc) < 2:
                    continue
                sim = SequenceMatcher(None, qc, tc).ratio()
                if qc in tc or tc in qc or sim > 0.60:
                    raw_results.append(
                        Candidate(
                            int((lefts[j] + widths[j] / 2) / current_sc),
                            int((tops[j] + heights[j] / 2) / current_sc),
                            float(max(sim, (conf / 100))),
                            f"ocr_m{p_num}",
                        ),
                    )
        return self._deduplicate_with_overlap_logic(raw_results)

    def _run_color_pass(
        self,
        r: NDArray,
        p: Path,
        tw: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run a standard BGR color-based template matching pass."""
        tpl, m, th = self._prep_tpl(p, tw)
        res = cv2.matchTemplate(
            cast("Any", r),
            cast("Any", tpl),
            cv2.TM_CCOEFF_NORMED,
            mask=cast("Any", m),
        )
        cb("PREP: Color Template", tpl, progress=15)
        return self._extract_tpl_locs(res, 0.7, tw, th, "tpl_color")

    def _run_lab_pass(
        self,
        r: NDArray,
        p: Path,
        tw: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run a template matching pass in the CIELAB color space for illumination invariance."""
        tpl_b, _, th = self._prep_tpl(p, tw)
        res = cv2.matchTemplate(
            cast("Any", cv2.cvtColor(cast("Any", r), cv2.COLOR_BGR2Lab)),
            cast("Any", cv2.cvtColor(cast("Any", tpl_b), cv2.COLOR_BGR2Lab)),
            cv2.TM_CCOEFF_NORMED,
        )
        cb("PREP: Lab Template", tpl_b, progress=20)
        return self._extract_tpl_locs(res, 0.7, tw, th, "tpl_lab")

    def _run_edge_pass(
        self,
        r: NDArray,
        p: Path,
        tw: int,
        cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run a template matching pass using Canny edge maps."""
        tpl_b, _, th = self._prep_tpl(p, tw)
        re = cv2.Canny(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), 50, 150)
        te = cv2.Canny(cv2.cvtColor(tpl_b, cv2.COLOR_BGR2GRAY), 50, 150)
        res = cv2.matchTemplate(cast("Any", re), cast("Any", te), cv2.TM_CCOEFF_NORMED)
        cb("PREP: Edge Map", re, progress=25)
        return self._extract_tpl_locs(res, 0.4, tw, th, "tpl_edge")

    def _run_orb_pass(
        self,
        r: NDArray,
        p: Path,
        _: int,
        _cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run an ORB feature-based matching pass to find keypoint clusters."""
        tpl = cast("Any", cv2.imread(str(p)))
        if tpl is None:
            return []
        orb = cv2.ORB_create(1000)
        _, d1 = orb.detectAndCompute(tpl, None)
        k2, d2 = orb.detectAndCompute(cast("Any", r), None)
        if d1 is None or d2 is None:
            return []
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        m = sorted(bf.match(d1, d2), key=lambda x: x.distance)
        if len(m) > 15:
            pts = np.float32(cast("Any", [k2[i.trainIdx].pt for i in m[:20]]))
            center = np.mean(pts, axis=0)
            return [Candidate(int(center[0]), int(center[1]), 0.9, "orb")]
        return []

    def _run_scaled_pass(
        self,
        r: NDArray,
        p: Path,
        tw: int,
        sf: float,
        _cb: Callable[..., Any],
    ) -> list[Candidate]:
        """Run a template matching pass with a specific scaling factor."""
        stw = int(tw * sf)
        tpl, m, th = self._prep_tpl(p, stw)
        if tpl.shape[0] > r.shape[0] or tpl.shape[1] > r.shape[1]:
            return []
        res = cv2.matchTemplate(
            cast("Any", r),
            cast("Any", tpl),
            cv2.TM_CCOEFF_NORMED,
            mask=cast("Any", m),
        )
        return self._extract_tpl_locs(res, 0.65, stw, th, f"scale_{sf}")

    def _prep_tpl(self, path: Path, tw: int) -> tuple[NDArray, NDArray | None, int]:
        """Prepare a template image by resizing and extracting alpha masks if available."""
        tpl_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        scale = tw / tpl_raw.shape[1]
        th = int(tpl_raw.shape[0] * scale)
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
        """Extract candidate coordinates and populate bounding boxes for geometry validation."""
        locs = np.where(res >= thr)
        candidates = []
        for pt in zip(*locs[::-1], strict=False):
            bbox = (int(pt[0]), int(pt[1]), tw, th)

            candidates.append(
                Candidate(
                    x=int(pt[0] + tw // 2),
                    y=int(pt[1] + th // 2),
                    score=float(res[pt[1], pt[0]]),
                    method=method,
                    bbox=bbox,
                ),
            )
        return candidates

    def _detect_desktop_icon_size(self, roi: NDArray, _cb: Callable[..., Any]) -> int:
        """Detect the dominant desktop icon size based on contour analysis."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sizes = [
            cv2.boundingRect(c)[2] for c in cnts if 30 < cv2.boundingRect(c)[2] < 150
        ]
        return int(max(set(sizes), key=sizes.count)) if sizes else 64

    def _draw_results(self, canvas: NDArray, candidates: list[Candidate]) -> NDArray:
        """Draw detection markers and labels onto the debug canvas."""
        viz = canvas.copy()
        for i, c in enumerate(candidates):
            color = (
                (0, 255, 0)
                if c.method == "fused_match"
                else (0, 255, 255)
                if "ocr" in c.method
                else (255, 255, 0)
            )
            cv2.drawMarker(
                cast("Any", viz),
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
        """Apply non-maximum suppression to remove overlapping template hits."""
        if not hits:
            return []
        sh = sorted(hits, key=lambda x: x.score, reverse=True)[:200]
        final: list[Candidate] = []
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
        func: Callable[..., list[Candidate]],
        *args: Any,  # noqa: ANN401
    ) -> list[Candidate]:
        """Wrap a function call to measure its execution time and log performance."""
        t0 = time.time()
        res = func(*args)
        self.perf_stats.append(PerfStat(name, (time.time() - t0) * 1000, len(res)))
        return res

    def _gui_benchmark_report(
        self,
        total_time: float,
        callback: Callable[[str, str, int | None], None] | None,
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

    def _apply_deep_ocr_preprocessing(self, r: NDArray, p_num: int) -> NDArray:
        """Apply a specific image preprocessing pipeline for deep OCR detection."""
        g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        if p_num == 1:
            return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if p_num == 2:
            return cv2.threshold(
                cv2.bitwise_not(g),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]
        if p_num == 5:
            return cv2.resize(g, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        if p_num in [11, 12]:
            up = cv2.resize(g, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            return cv2.threshold(
                cv2.morphologyEx(up, cv2.MORPH_TOPHAT, k),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]
        if p_num == 8:
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
        """Deduplicate candidate results by filtering out overlapping hits within a radius."""
        clean: list[Candidate] = []
        for cand in sorted(raw_results, key=lambda x: x.score, reverse=True):
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
        """Finalize detection results by fusing template and OCR hits into a single list."""
        final_list: list[Candidate] = []
        used_ocr, used_tpl = set(), set()
        for i, t in enumerate(t_hits):
            for j, o in enumerate(o_hits):
                if j in used_ocr:
                    continue
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
        final_list.extend([t for i, t in enumerate(t_hits) if i not in used_tpl])
        final_list.extend([o for j, o in enumerate(o_hits) if j not in used_ocr])
        res: list[Candidate] = []
        for c in sorted(final_list, key=lambda x: x.score, reverse=True):
            if not any(
                np.sqrt((c.x - r.x) ** 2 + (c.y - r.y) ** 2) < ts * 0.7 for r in res
            ):
                res.append(c)
        return [r for r in res if r.score >= thr]

    def _validate_spatial_consistency(
        self,
        icon_path: Path,
        hits: list[Candidate],
    ) -> list[Candidate]:
        """Filter candidates that deviate significantly from the template's aspect ratio."""
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

                deviation = min(target_ratio, current_ratio) / max(
                    target_ratio,
                    current_ratio,
                )
                hit.geometry_score = float(deviation)

                hit.score *= 0.8 + (0.2 * deviation)

            validated_hits.append(hit)

        return validated_hits

    def _enhance_contrast_adaptive(self, img: NDArray) -> NDArray:
        """Apply adaptive histogram equalization to improve text visibility."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        channels = list[NDArray](cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        channels[0] = clahe.apply(channels[0])
        limg = cv2.merge(cast("NDArray", channels))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
