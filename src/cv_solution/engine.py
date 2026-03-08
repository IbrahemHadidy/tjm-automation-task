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

from cv_solution.constants import RECOVERY_QUEUE_LIMIT
from cv_solution.models import Candidate, DetectionMethod, GroundingConfig, PerfStat
from cv_solution.processors.fusion import FusionProcessor
from cv_solution.processors.ocr import OCRProcessor
from cv_solution.processors.visual import VisualProcessor
from cv_solution.utils import ImageUtils, set_high_dpi_awareness

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
        set_high_dpi_awareness()
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
            priority: The strategy to use ('fusion', 'text', or 'score').

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
            # Target both global and recovery OCR methods via Enum
            ocr_methods = {DetectionMethod.OCR_GLOBAL, DetectionMethod.OCR_RECOVERY}
            ocr_cands = [c for c in candidates if c.method in ocr_methods]
            if ocr_cands:
                return max(ocr_cands, key=lambda c: c.score)

        # Default fallback: highest absolute score regardless of method
        return max(candidates, key=lambda c: c.score)

    def locate_elements(
        self,
        screenshot: Image.Image,
        icon_image: Path | None,
        text_query: str,
        config: GroundingConfig | None = None,
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Locate UI elements on screen by executing a multi-stage vision pipeline.

        Workflow:
            1. Convert PIL screenshot to OpenCV BGR.
            2. Execute Visual Template Matching passes.
            3. Execute Global OCR sweep.
            4. Apply NMS and Geometric validation to visual hits.
            5. Perform Targeted OCR recovery on remaining visual anchors.
            6. Fuse and filter all evidence into final candidates.

        Args:
            screenshot: The raw desktop capture in PIL format.
            icon_image: Path to the template image to find.
            text_query: String label to search for via OCR.
            config: Configuration overrides for thresholds and scaling.
            callback: Optional logger for progress and debug frames.

        Returns:
            A list of detected Candidate objects sorted by confidence.

        """
        self.perf_stats = []
        safe_config = config or GroundingConfig()
        t0 = time.time()

        # 1. Image Conversion & ROI Setup
        full_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        self.last_raw_frame = full_img.copy()
        desktop_roi = ImageUtils.crop_to_desktop(full_img)

        self._log("INIT: Screenshot processed", desktop_roi, callback, progress=5)

        if self.should_abort():
            return []

        # 2. Heuristic Detection
        target_size = ImageUtils.detect_icon_size(desktop_roi, safe_config)

        # 3. Probabilistic Features
        template_hits = self._run_template_passes(
            desktop_roi,
            icon_image,
            target_size,
            safe_config,
            callback,
        )
        ocr_hits = self._run_ocr(desktop_roi, text_query, safe_config, callback)

        # 4. Refine Visual Evidence
        if self.fusion_processor is None:
            self.fusion_processor = FusionProcessor(safe_config)

        templates = self.fusion_processor.apply_nms(template_hits, target_size)
        if icon_image:
            templates = self.fusion_processor.validate_geometry(icon_image, templates)

        # 5. Targeted Recovery (Contextual OCR)
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

        # 6. Final Fusion & Confidence Filtering
        final_candidates = self.fusion_processor.fuse_and_filter(
            templates,
            ocr_hits,
            target_size,
        )

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
                cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if len(frame.shape) == 2
                else frame.copy()
            )

        if callback:
            callback(msg, lvl, progress)

    def _run_template_passes(
        self,
        roi: MatLike,
        icon: Path | None,
        target_size: int,
        config: GroundingConfig,
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Delegate visual detection to the VisualProcessor.

        Args:
            roi: The image region to search.
            icon: Path to the icon template.
            target_size: The base size of UI elements on the current desktop.
            config: Grounding configuration.
            callback: Logging callback.

        Returns:
            List of candidates found via template matching.

        """
        if not icon:
            return []

        self.visual_processor = self.visual_processor or VisualProcessor(config)

        hits = self.visual_processor.run_all(
            roi=roi,
            icon_path=icon,
            target_size=target_size,
            log_callback=lambda m, f=None, **kwargs: self._log(
                m,
                f,
                callback,
                progress=kwargs.get("progress"),
            ),
            should_abort=self.should_abort,
        )

        self.perf_stats.extend(self.visual_processor.last_stats)
        return hits

    def _run_ocr(
        self,
        roi: MatLike,
        text_query: str,
        config: GroundingConfig,
        callback: LogCallback | None = None,
    ) -> list[Candidate]:
        """Run global OCR sweep using the OCRProcessor.

        Args:
            roi: The full desktop image region.
            text_query: Text to search for.
            config: Grounding configuration.
            callback: Logging callback.

        Returns:
            List of candidates found via global text search.

        """
        if not text_query or not config.use_ocr:
            return []

        t0 = time.time()
        self.ocr_processor = self.ocr_processor or OCRProcessor(
            self.tesseract_path,
            config,
        )

        hits = self.ocr_processor.search_global(
            roi=roi,
            query=text_query,
            log_callback=lambda m, f=None, **kwargs: self._log(
                m,
                f,
                callback,
                progress=kwargs.get("progress"),
            ),
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
        """Perform high-resolution local OCR around high-probability visual anchors.

        Args:
            roi: The full desktop image region.
            templates: Visual anchors used to define recovery ROIs.
            text_query: Text to verify within the ROIs.
            target_size: Base size for ROI calculation.
            config: Grounding configuration.
            callback: Logging callback.

        Returns:
            List of candidates found via targeted recovery.

        """
        if not templates or not text_query:
            return []

        queue = templates[:RECOVERY_QUEUE_LIMIT]
        self._log(
            f"RECOVERY: Verifying {len(queue)} anchors",
            callback=callback,
            progress=82,
        )

        t0 = time.time()
        self.ocr_processor = self.ocr_processor or OCRProcessor(
            self.tesseract_path,
            config,
        )

        hits = self.ocr_processor.recover_labels(
            roi,
            queue,
            text_query,
            target_size,
            lambda m, f: self._log(m, f, callback=callback),
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
            header = f"| {'ID':<4} | {'Method':<15} | {'Score':<8} | {'Coords':<15} |"
            divider = f"|{'-' * 6}|{'-' * 17}|{'-' * 10}|{'-' * 17}|"
            table = ["### Candidate Detection Summary", header, divider]

            for i, c in enumerate(candidates):
                coords = f"({c.x}, {c.y})"
                method_name = (
                    c.method.value if hasattr(c.method, "value") else str(c.method)
                )
                table.append(
                    f"| {i + 1:<4} | {method_name:<15} | {c.score:<8.2f} | {coords:<15} |",
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
        report.append(f"| {'Pass Name':<20} | {'Time (ms)':<10} | {'Hits':<8} |")
        report.append(f"|{'-' * 22}|{'-' * 12}|{'-' * 10}|")

        report.extend(
            [
                f"| {stat.name:<20} | {stat.duration_ms:<10.0f} | {stat.items_found:<8} |"
                for stat in self.perf_stats
            ],
        )
        callback("\n".join(report), "HEAD", 100)
