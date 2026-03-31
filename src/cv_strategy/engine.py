"""CVGroundingEngine.

Provides computer vision and OCR-based desktop UI element grounding using
a multi-pass probabilistic detection pipeline. This engine orchestrates
visual template matching, global OCR sweeps, and targeted local recovery.
"""

import time
from typing import TYPE_CHECKING, ParamSpec

import cv2
import numpy as np
import pytesseract

from cv_strategy.constants import GRAY_TO_BGR, RECOVERY_QUEUE_LIMIT, RGB_TO_BGR
from cv_strategy.models import Candidate, DetectionMethod, GroundingConfig, PerfStat
from cv_strategy.processors.fusion import FusionProcessor
from cv_strategy.processors.ocr import OCRProcessor
from cv_strategy.processors.visual import VisualProcessor
from cv_strategy.utils import ImageUtils

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cv2.typing import MatLike
    from PIL import Image


P = ParamSpec("P")

# A callback signature used by the engine to emit logs and intermediate visualizations.
type LogCallback = Callable[[str, str, int | None], None]


class CVGroundingEngine:
    """Orchestrate multi-modal searches for UI elements using OpenCV and Tesseract.

    This class serves as the central entry point for the grounding pipeline,
    managing the lifecycle of specialized processors and aggregating results.
    """

    def __init__(self, tesseract_path: str) -> None:
        """Initialize the engine and set global Tesseract configuration.

        Args:
            tesseract_path: Absolute path to the Tesseract executable.

        Raises:
            RuntimeError: If high DPI awareness cannot be set.

        """
        self.tesseract_path: str = tesseract_path
        # Set the global path for the pytesseract wrapper
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        # Lazy-loaded processors
        self.ocr_processor: OCRProcessor | None = None
        self.visual_processor: VisualProcessor | None = None
        self.fusion_processor: FusionProcessor | None = None

        self.last_raw_frame: MatLike | None = None
        self.last_debug_frame: MatLike | None = None
        self.perf_stats: list[PerfStat] = []
        self.should_abort: Callable[[], bool] = lambda: False

    def select_best_candidate(
        self,
        candidates: list[Candidate],
        priority: str = "fusion",
    ) -> Candidate | None:
        """Select the highest-confidence candidate using a priority-based strategy.

        Args:
            candidates: List of candidates returned by locate_elements.
            priority: The strategy to use ('fusion', 'text', 'vision', or 'score').

        Returns:
            The best matching Candidate or None if the list is empty.

        """
        if not candidates:
            return None

        if priority == "fusion":
            fused = [c for c in candidates if c.method == DetectionMethod.FUSED]
            if fused:
                return max(fused, key=lambda c: c.score)

        if priority == "text":
            ocr_methods = {DetectionMethod.OCR_GLOBAL, DetectionMethod.OCR_RECOVERY}
            ocr_cands = [c for c in candidates if c.method in ocr_methods]
            if ocr_cands:
                return max(ocr_cands, key=lambda c: c.score)

        if priority == "vision":
            vision_methods = {
                DetectionMethod.COLOR,
                DetectionMethod.LAB,
                DetectionMethod.EDGE,
                DetectionMethod.GRAY,
                DetectionMethod.SCALE,
            }
            vision_cands = [c for c in candidates if c.method in vision_methods]
            if vision_cands:
                return max(vision_cands, key=lambda c: c.score)

        # Default fallback: highest absolute score regardless of method
        return max(candidates, key=lambda c: c.score)

    def locate_elements(
        self,
        screenshot: Image.Image,
        icon_path: Path | None,
        text_query: str,
        config: GroundingConfig | None = None,
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Locate UI elements using a multi-scale, size-aware vision pipeline.

        Workflow:
            1. Analyze ROI to detect all potential icon size clusters (e.g., 32px, 48px).
            2. Run visual template matching for EVERY detected size cluster.
            3. Execute global OCR sweep for text-first anchors.
            4. Apply 'Smart NMS' which deduplicates hits using local scaling.
            5. Perform Targeted OCR Recovery using search windows relative to
               each icon's specific detected scale.
            6. Fuse visual and text evidence into final weighted candidates.

        Args:
            screenshot: The raw desktop capture in PIL format.
            icon_path: Path to the template image to find.
            text_query: String label to search for via OCR.
            config: Configuration overrides for thresholds and recovery.
            callback: Optional logger for progress and debug frames.

        Returns:
            A list of detected Candidate objects, where each candidate contains
            the specific 'base_size' it was matched against.

        """
        self.perf_stats = []
        safe_config = config or GroundingConfig()
        t0 = time.time()

        # 1. Image Conversion
        full_img = cv2.cvtColor(np.array(screenshot), RGB_TO_BGR)
        self.last_raw_frame = full_img.copy()
        desktop_roi = ImageUtils.crop_to_desktop(full_img)

        self._log("INIT: Screenshot processed", desktop_roi, callback, progress=5)

        if self.should_abort():
            return []

        # 2. Size Detection
        candidate_sizes = ImageUtils.detect_icon_sizes(desktop_roi, safe_config)

        # 3. Visual Search (Casts a wide net across all potential scales)
        template_hits = []
        if icon_path:
            for size in candidate_sizes:
                hits = self._run_template_passes(
                    desktop_roi,
                    icon_path,
                    size,
                    safe_config,
                    callback,
                )
                # IMPORTANT: Inside _run_template_passes, each hit should
                # now have c.extra["base_size"] = size
                template_hits.extend(hits)

        # 4. Global OCR Sweep
        ocr_hits = self._run_ocr(desktop_roi, text_query, safe_config, callback)

        # 5. FIXED: Size-Aware Refinement
        if self.fusion_processor is None:
            self.fusion_processor = FusionProcessor(safe_config)

        # We no longer pass a single target_size.
        # The processor must now look at Candidate.extra["base_size"]
        templates = self.fusion_processor.apply_smart_nms(template_hits)

        if icon_path:
            templates = self.fusion_processor.validate_geometry(icon_path, templates)

        # 6. FIXED: Recovery using candidate-specific sizes
        if safe_config.enable_recovery:
            ocr_hits.extend(
                self._targeted_recovery(
                    desktop_roi,
                    templates,
                    text_query,
                    # We pass the default as a fallback only
                    candidate_sizes[0],
                    safe_config,
                    callback,
                ),
            )

        # 7. Final Fusion (Filtering logic now uses local candidate scale)
        final_candidates = self.fusion_processor.fuse_and_filter(templates, ocr_hits)

        self._report_results(self.last_raw_frame.copy(), final_candidates, t0, callback)
        return final_candidates

    def _log(
        self,
        msg: str,
        frame: MatLike | None = None,
        callback: LogCallback | None = None,
        lvl: str = "INFO",
        progress: int | None = None,
    ) -> None:
        """Route internal logging for engine progress and telemetry.

        Args:
            msg: Descriptive message to log.
            frame: Optional OpenCV image to store for debugging.
            callback: The UI or console callback to emit the message to.
            lvl: Logging level (e.g., 'INFO', 'HEAD').
            progress: Optional integer representing pipeline progress (0-100).

        """
        if self.should_abort():
            return

        if frame is not None:
            # Normalize frame for debug storage
            self.last_debug_frame = (
                cv2.cvtColor(frame, GRAY_TO_BGR)
                if len(frame.shape) == 2
                else frame.copy()
            )

        if callback:
            callback(msg, lvl, progress)

    def _run_template_passes(
        self,
        roi: MatLike,
        icon_path: Path | None,
        target_size: int,
        config: GroundingConfig,
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Delegate visual detection to the VisualProcessor.

        Args:
            roi: The image region to search.
            icon_path: Path to the icon template.
            target_size: The base size of UI elements on the current desktop.
            config: Grounding configuration.
            callback: Logging callback.

        Returns:
            List of candidates found via template matching.

        """
        if not icon_path:
            return []

        self.visual_processor = self.visual_processor or VisualProcessor(config)

        hits = self.visual_processor.run_all(
            roi=roi,
            icon_path=icon_path,
            target_size=target_size,
            log_callback=lambda m, f=None, **kwargs: self._log(
                m,
                f,
                callback,
                progress=kwargs.get("progress"),
            ),
            should_abort=self.should_abort,
        )

        for hit in hits:
            hit.extra["base_size"] = target_size

        return hits

    def _run_ocr(
        self,
        roi: MatLike,
        text_query: str,
        config: GroundingConfig,
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Execute a global OCR sweep over the entire desktop region.

        Uses the updated OCRProcessor to return **all matches** for the query,
        including exact, substring, and fuzzy hits.

        Args:
            roi: The full desktop image region in OpenCV BGR format.
            text_query: Text string to search for.
            config: Grounding configuration, including thresholds and flags.
            callback: Optional logger for progress updates and debug frames.

        Returns:
            List[Candidate]: OCR candidates detected across the full desktop.

        """
        if not text_query or not config.use_ocr:
            return []

        t0 = time.time()
        self.ocr_processor = self.ocr_processor or OCRProcessor(
            self.tesseract_path,
            config,
        )

        # Run global OCR with query matching and return all hits
        hits = self.ocr_processor.search_global(
            roi=roi,
            query=text_query,
            log_callback=callback or (lambda *_args, **_kwargs: None),
        )

        self.perf_stats.append(
            PerfStat("OCR Global Sweep", (time.time() - t0) * 1000, len(hits)),
        )
        self._log("OCR Sweep Completed", callback=callback, progress=80)
        return hits

    def _targeted_recovery(
        self,
        roi: MatLike,
        templates: list[Candidate],
        text_query: str,
        target_size: int,
        config: GroundingConfig,
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Perform high-resolution OCR around high-probability visual anchors.

        Updated to use the OCRProcessor recovery mode that returns multiple
        matches per anchor. Uses threading internally for efficiency.

        Args:
            roi: Full desktop image region in OpenCV BGR format.
            templates: Visual anchor candidates used to define recovery ROIs.
            text_query: Text string to verify within the cropped ROIs.
            target_size: Base size for determining crop regions.
            config: Grounding configuration, including thresholds and flags.
            callback: Optional logger for progress updates and debug frames.

        Returns:
            List[Candidate]: OCR candidates detected in the localized regions.

        """
        if not templates or not text_query:
            return []

        # Limit number of anchors to prevent overload
        queue = templates[: min(RECOVERY_QUEUE_LIMIT, len(templates))]

        t0 = time.time()
        self.ocr_processor = self.ocr_processor or OCRProcessor(
            self.tesseract_path,
            config,
        )

        hits = self.ocr_processor.recover_labels(
            img=roi,
            templates=queue,
            query=text_query,
            target_size=target_size,
            log_callback=callback or (lambda *_args, **_kwargs: None),
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
        """Handle final visualization and telemetry reporting.

        Args:
            canvas: The original image to draw results upon.
            candidates: Final filtered list of candidates.
            t0: Start timestamp for total duration calculation.
            callback: Reporting callback.

        """
        self.last_debug_frame = ImageUtils.draw_candidates(canvas, candidates)
        duration_ms = (time.time() - t0) * 1000

        self._gui_benchmark_report(duration_ms, callback)

        if callback and candidates:
            # Table definitions with consistent padding
            header = f"| {'ID':<4} | {'Method':<25} | {'Score':<8} | {'Coords':<15} |"
            divider = f"|{'-' * 6}|{'-' * 27}|{'-' * 10}|{'-' * 17}|"
            table = ["### Candidate Detection Summary", header, divider]

            for i, c in enumerate(candidates):
                coords = f"({c.x}, {c.y})"

                # Safe method name resolution
                method_name = getattr(c.method, "value", str(c.method))

                # Logic for fused naming
                if "fused_match" in method_name:
                    v_source = c.extra.get("visual_source")
                    v_name = (
                        getattr(v_source, "value", str(v_source))
                        if v_source
                        else "unknown"
                    )
                    method_name = f"fused ({v_name})"

                table.append(
                    f"| {i + 1:<4} | {method_name:<25} | {c.score:<8.2f} | {coords:<15} |",
                )

            callback("\n".join(table), "INFO", 100)

        self._log(
            f"FINISH: Found {len(candidates)} candidates in {duration_ms:.0f}ms",
            self.last_debug_frame,
            callback=callback,
            progress=100,
        )

    def _gui_benchmark_report(
        self,
        total_time: float,
        callback: LogCallback | None,
    ) -> None:
        """Construct and emit a performance benchmark report.

        Args:
            total_time: Total execution time in milliseconds.
            callback: Reporting callback.

        """
        if not callback:
            return

        report = [f"### VISION ENGINE BENCHMARK ({total_time:.0f}ms)"]
        report.append(f"| {'Pass Name':<30} | {'Time (ms)':<12} | {'Hits':<8} |")
        report.append(f"|{'-' * 32}|{'-' * 14}|{'-' * 10}|")

        report.extend(
            [
                f"| {stat.name:<30} | {stat.duration_ms:<12.0f} | {stat.items_found:<8} |"
                for stat in self.perf_stats
            ],
        )
        callback("\n".join(report), "HEAD", 100)
