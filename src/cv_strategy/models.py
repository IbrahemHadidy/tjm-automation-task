"""Define data models and configuration containers for the grounding engine.

Provide structured schemas for detection candidates, performance metrics,
and engine configuration settings used throughout the CV pipeline.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from cv_strategy.constants import ICON_MAX_WIDTH, ICON_MIN_WIDTH

if TYPE_CHECKING:
    from cv2.typing import MatLike


class DetectionMethod(StrEnum):
    """Enumeration of detection strategies to avoid string-based errors.

    Attributes:
        COLOR: Standard BGR template matching.
        LAB: Perceptual color space matching (lighting robust).
        EDGE: Canny edge-based structural matching.
        GRAY: Grayscale intensity matching.
        SCALE: Multiscale template matching.
        OCR_GLOBAL: Initial full-screen text search.
        OCR_RECOVERY: Targeted local area text verification.
        FUSED: Final result after Non-Maximum Suppression (NMS).

    """

    COLOR = "tpl_color"
    LAB = "tpl_lab"
    EDGE = "tpl_edge"
    GRAY = "tpl_gray"
    SCALE = "tpl_scale"
    OCR_GLOBAL = "ocr_global"
    OCR_RECOVERY = "ocr_recovery"
    FUSED = "fused_match"


@dataclass(frozen=True)
class Candidate:
    """Represent a potential UI element location identified by the engine.

    Attributes:
        x: Horizontal center coordinate of the detection.
        y: Vertical center coordinate of the detection.
        score: Aggregated confidence score (typically 0.0 to 1.0+).
        method: The specific detection strategy used to find this candidate.
        img_score: Raw visual similarity score from template matching.
        txt_score: Raw OCR similarity score from text recognition.
        bbox: Bounding box tuple (x, y, w, h) defining the element's area.
        geometry_score: Multiplier based on aspect ratio or spatial consistency.
        extra: Metadata dictionary for pass-specific info (e.g., scale, text).

    """

    x: int
    y: int
    score: float
    method: DetectionMethod
    img_score: float = 0.0
    txt_score: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    geometry_score: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerfStat:
    """Store execution metrics for a specific detection or processing pass.

    Attributes:
        name: Human-readable identifier of the operation.
        duration_ms: Total execution time measured in milliseconds.
        items_found: Count of raw candidates generated during this pass.

    """

    name: str
    duration_ms: float
    items_found: int


@dataclass
class GroundingResult:
    """Encapsulate the output of a complete grounding operation.

    Attributes:
        candidates: Sorted list of ranked and deduplicated results.
        perf_stats: Collection of timing data for internal engine passes.
        debug_frame: Annotated BGR image visualizing detection markers.
        total_time_ms: Total wall-clock time elapsed for the grounding call.

    """

    candidates: list[Candidate]
    perf_stats: list[PerfStat] = field(default_factory=list)
    debug_frame: MatLike | None = None
    total_time_ms: float = 0.0


@dataclass(frozen=True)
class GroundingConfig:
    """Configure the behavior of the CVGroundingEngine pipeline.

    Attributes:
        use_color: Toggle standard BGR template matching.
        use_lab: Toggle CIELAB color space matching for lighting invariance.
        use_edge: Toggle Canny edge-based template matching.
        use_gray: Toggle grayscale intensity matching.
        use_ocr: Toggle global OCR text search.
        use_multiscale: Toggle searching for icons across multiple scales.
        use_adaptive: Toggle CLAHE contrast enhancement for input frames.
        enable_recovery: Toggle targeted recovery OCR around visual anchors.
        num_cores: Set maximum threads for parallel detection passes.
        ocr_lang: Define Tesseract language code (e.g., 'eng').
        threshold: Set minimum confidence score required to return a candidate.
        psm: Set Tesseract Page Segmentation Mode.
        min_icon_width: Set floor for icon scale auto-detection.
        max_icon_width: Set ceiling for icon scale auto-detection.

    """

    # Detection Toggles
    use_color: bool = True
    use_lab: bool = True
    use_edge: bool = True
    use_gray: bool = True
    use_ocr: bool = True
    use_multiscale: bool = True
    use_adaptive: bool = False
    enable_recovery: bool = True

    # Engine Settings
    num_cores: int = 8
    ocr_lang: str = "eng"

    # Engine Sensitivity & OCR behavior
    threshold: float = 0.7
    psm: int = 11

    # Icon Detection Bounds
    min_icon_width: int = ICON_MIN_WIDTH
    max_icon_width: int = ICON_MAX_WIDTH
