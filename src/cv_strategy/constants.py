"""Centralize configuration constants for computer vision and OCR engines.

This module defines the thresholds, scaling factors, and scoring weights used
across the detection pipeline. All 'FACTOR' constants are relative to the
target template size (e.g., 0.5 = 50% of icon width/height).

Design principles:
- Uppercase variables are treated as constants by convention.
- Values marked with `Final` are true invariants and must not change.
- Non-Final values are intentionally tunable and may be adjusted during calibration.
- This module is the single source of truth for all pipeline tuning.
"""

from enum import IntEnum
from typing import Final

# ============================================
# Tesseract Engine Settings
# Global configuration for the Tesseract binary and underlying OCR engine behavior.
# ============================================

OCR_ENGINE_MODE: Final[int] = 3
"""Engine Mode (OEM) 3: Default, based on what is available (Legacy + LSTM)."""

PSM_SINGLE_LINE: Final[int] = 7
"""Page Segmentation Mode (PSM) 7: Treat the image as a single text line (best for isolated labels)."""

PSM_AUTO: Final[int] = 3
"""Page Segmentation Mode (PSM) 3: Fully automatic page segmentation (best for full-screen search)."""

# ============================================
# OCR Thresholds & Scoring
# Constants used to filter, merge, and score text results extracted by the OCR engine.
# ============================================

OCR_MIN_CONFIDENCE: int = 10
"""Minimum Tesseract confidence (0-100) to accept a token.

Typical range: 0-40.
Lower values increase recall; higher values reduce noisy tokens.
"""

OCR_MIN_TOKEN_LENGTH: int = 2
"""Minimum string length to accept a token.

Typical range: 1-3.
Use 1 for terse labels, 2 for general UI text, and 3 for stricter filtering.
"""

OCR_RECOVERY_THRESHOLD: float = 0.5
"""Minimum similarity score (0.0-1.0) for ROI recovery.

Typical range: 0.40-0.65.
Lower values are more permissive; higher values are safer but may miss matches.
"""

OCR_FUZZY_MATCH_THRESHOLD: float = 0.6
"""Minimum similarity score (0.0-1.0) for fuzzy matching.

Typical range: 0.50-0.75.
Use lower values when OCR noise is expected; use higher values for stricter matching.
"""

OCR_EXACT_MATCH_SCORE: float = 1.0
"""Score awarded for perfect string equality.

Fixed reference value; normally left unchanged.
"""

OCR_SUBSTRING_BASE_SCORE: float = 0.5
"""Starting point score if the query is found inside a longer string.

Typical range: 0.30-0.70.
Higher values favor substring hits more strongly.
"""

OCR_SUBSTRING_MATCH_WEIGHT: float = 0.35
"""Multiplier for the length ratio of the match in substring logic.

Typical range: 0.20-0.50.
Higher values reward longer matches more aggressively.
"""

OCR_SIMILARITY_WEIGHT: float = 0.5
"""Weight for Levenshtein distance in fuzzy mode scoring.

Typical range: 0.30-0.70.
Lower values make fuzzy scoring more conservative; higher values make it more permissive.
"""

OCR_RECOVERY_PENALTY: float = 0.05
"""Penalty applied to recovery matches to ensure Global Search results take precedence.

Typical range: 0.02-0.15.
Higher penalties reduce recovery preference more aggressively.
"""


class OCRPreprocessingMode(IntEnum):
    """Internal mode IDs for the image preprocessing pipeline."""

    OTSU = 1
    INVERTED_OTSU = 2
    CUBIC_UPSCALE = 5
    GRAY = 8
    TOPHAT_UPSCALE_A = 11
    TOPHAT_UPSCALE_B = 12


OCR_GLOBAL_SEARCH_MODES: tuple[OCRPreprocessingMode, ...] = (
    OCRPreprocessingMode.OTSU,
    OCRPreprocessingMode.INVERTED_OTSU,
    OCRPreprocessingMode.CUBIC_UPSCALE,
    OCRPreprocessingMode.GRAY,
    OCRPreprocessingMode.TOPHAT_UPSCALE_A,
    OCRPreprocessingMode.TOPHAT_UPSCALE_B,
)
"""Modes executed during the initial full-screen 'Global' search pass."""

# ============================================
# Image Preprocessing & Scaling
# Constants for color space conversions, resizing, and morphological operations.
# ============================================

MAX_8BIT_VALUE: Final[int] = 255
"""Standard bit-depth maximum for 8-bit grayscale images."""

INTERPOLATION_UP: Final[int] = 3
"""cv2.INTER_CUBIC: Best for upscaling."""

INTERPOLATION_LOCAL: Final[int] = 1
"""cv2.INTER_LINEAR: Standard resizing."""

BGR_TO_RGB: Final[int] = 4
"""cv2.COLOR_BGR2RGB."""

BGR_TO_GRAY: Final[int] = 6
"""cv2.COLOR_BGR2GRAY."""

BGR_TO_LAB: Final[int] = 44
"""cv2.COLOR_BGR2LAB."""

RGB_TO_BGR: Final[int] = 4
"""cv2.COLOR_RGB2BGR."""

GRAY_TO_BGR: Final[int] = 8
"""cv2.COLOR_GRAY2BGR."""

LAB_TO_BGR: Final[int] = 56
"""cv2.COLOR_LAB2BGR."""

OCR_GLOBAL_UPSCALE_FACTOR: float = 2.5
"""Multiplier for image resolution enhancement before global OCR.

Typical range: 1.5-3.0.
Higher values can improve tiny text recognition but may blur fine structure after interpolation.
"""

OCR_LOCAL_UPSCALE_FACTOR: float = 2.0
"""Multiplier for image resolution enhancement before local OCR.

Typical range: 1.25-2.5.
Usually slightly lower than the global upscale factor to preserve local detail.
"""

OCR_MORPH_KERNEL_SIZE: tuple[int, int] = (9, 9)
"""Kernel size for morphological Top-hat filtering to isolate text.

Typical range: (5, 5) to (15, 15).
Smaller kernels preserve more detail; larger kernels suppress broader background structure.
"""

# ============================================
# Visual Engine Technical Settings
# Core algorithmic parameters used for template alignment.
# ============================================

TPL_MATCH_METHOD: Final[int] = 5
"""Template Matching Method (5 = cv2.TM_CCOEFF_NORMED)."""

# ============================================
# Geometry & Aspect Ratio Validation
# Logic and weights for adjusting visual match scores based on shape consistency.
# ============================================

GEOM_BASE_SCORE_WEIGHT: float = 0.8
"""Trust floor for any detected shape.

Typical range: 0.70-0.90.
Higher values preserve the raw match score more strongly; lower values dampen it harder.
"""

GEOM_RATIO_BONUS_WEIGHT: float = 0.2
"""Extra confidence for perfect aspect ratio matches.

Typical range: 0.10-0.30.
Higher values reward shape consistency more strongly.
"""

# ============================================
# Template Matching Thresholds
# Sensitivity settings for the different CV matching algorithms.
# ============================================

TPL_COLOR_THRESHOLD: float = 0.7
"""Strictness threshold for color-based template matching.

Typical range: 0.60-0.85.
Lower values increase recall; higher values reduce false positives.
"""

TPL_LAB_THRESHOLD: float = 0.7
"""Strictness threshold for LAB space similarity.

Typical range: 0.60-0.85.
Useful when perceptual color similarity is more important than raw channel similarity.
"""

TPL_EDGE_THRESHOLD: float = 0.4
"""Lower threshold for Canny edge matching due to high sensitivity.

Typical range: 0.25-0.55.
Edge maps are sparse and noisy, so this threshold is usually lower than color or gray matching.
"""

TPL_GRAY_THRESHOLD: float = 0.65
"""Strictness threshold for grayscale template matching.

Typical range: 0.55-0.80.
This is often a good middle-ground threshold for UI icon detection.
"""

TPL_MULTISCALE_THRESHOLD: float = 0.65
"""Strictness threshold for multiscale matching passes.

Typical range: 0.55-0.80.
Multiscale searches are more permissive than the base match because resizing introduces interpolation noise.
"""

CANNY_LOW_THRESHOLD: int = 50
"""Low threshold for Canny edge detection.

Typical range: 30-100.
Lower values detect more weak edges; higher values suppress more noise.
"""

CANNY_HIGH_THRESHOLD: int = 150
"""High threshold for Canny edge detection.

Typical range: 100-250.
Higher values require stronger gradients before an edge is accepted.
"""

MULTISCALE_FACTORS: tuple[float, float] = (0.8, 1.25)
"""Scale factors used during multiscale template matching.

Typical range: 0.75-1.50 for desktop icon work.
The values should bracket the expected icon-size drift without exploding runtime.
"""

MAX_TEMPLATE_HITS: int = 200
"""Limit raw hits per method to prevent NMS performance degradation.

Typical range: 50-500 depending on image size and threshold strictness.
Lower values improve speed; higher values preserve more candidate density.
"""

# ============================================
# Detection Fusion & Deduplication
# Post-processing rules for merging overlapping detections.
# ============================================

NMS_IOU_THRESHOLD: float = 0.30
"""Intersection-over-Union (IoU) threshold used for Non-Maximum Suppression (NMS).

Typical range: 0.25-0.40 for UI icon detection.
- Lower values (e.g., 0.20): More aggressive suppression.
- Higher values (e.g., 0.40): More candidates retained.
"""

NMS_RADIUS_FACTOR: float = 0.6
"""Deduplication radius for intra-method overlap removal.

Typical range: 0.40-0.80.
This is a legacy-style spatial heuristic and is less precise than IoU-based suppression.
"""

FUSION_DISTANCE_FACTOR: float = 1.5
"""Maximum distance to pair a visual hit with an OCR label.

Typical range: 1.0-2.0.
Higher values allow looser pairing between visual and OCR detections.
"""

FINAL_DEDUP_RADIUS_FACTOR: float = 0.7
"""Final cross-method cleanup radius factor.

Typical range: 0.50-0.90.
Use lower values for stricter deduplication and higher values for more permissive clustering.
"""

MIN_DEDUPE_RADIUS_FACTOR: float = 0.8
"""Safety floor for deduplication radii.

Typical range: 0.60-1.00.
Prevents deduplication from becoming too aggressive when calibration inputs are small.
"""

FUSION_SCORE_BONUS: float = 0.1
"""Bonus added when both Visual AND OCR confirm the same element.

Typical range: 0.05-0.20.
Higher values reward cross-modal agreement more strongly.
"""

# ============================================
# Recovery Search Geometry
# Multipliers defining the spatial boundaries when searching for text near an icon.
# ============================================

RECOVERY_HORIZONTAL_PAD_FACTOR: float = 0.4
"""Width extension for text search relative to icon center.

Typical range: 0.25-0.60.
Higher values widen the search region horizontally around the icon.
"""

RECOVERY_VERTICAL_EXTEND_FACTOR: float = 1.6
"""Height extension for recovery search.

Typical range: 1.2-2.0.
Labels are usually below the icon, so vertical expansion is often asymmetric.
"""

RECOVERY_VERTICAL_OFFSET_PX: int = 5
"""Slight downward shift for the search ROI in pixels.

Typical range: 0-10 px.
Useful when labels tend to sit a few pixels below the icon.
"""

# ============================================
# Desktop Detection & Performance
# Environmental constraints and limits to maintain real-time performance.
# ============================================

TASKBAR_HEIGHT_PX: int = 60
"""Standard height of the Windows taskbar.

Typical range: 40-80 px depending on DPI and taskbar configuration.
"""

ICON_MIN_WIDTH: int = 30
"""Minimum pixel width for a valid desktop icon candidate.

Typical range: 16-48 px.
Useful for filtering out tiny noise and non-icon UI fragments.
"""

ICON_MAX_WIDTH: int = 150
"""Maximum pixel width for a valid desktop icon candidate.

Typical range: 96-192 px.
Helps exclude windows, panels, and other large non-icon regions.
"""

DEFAULT_ICON_SIZE: int = 64
"""Default target size for icon templates.

Typical range: 32-96 px.
A middle value works well for standard desktop icon workflows.
"""

RECOVERY_QUEUE_LIMIT: int = 12
"""Limit on sub-regions processed per frame to maintain speed.

Typical range: 8-24.
Lower values keep the pipeline responsive; higher values improve coverage at a cost.
"""

# ============================================
# Debug Visualization Settings (BGR)
# Colors, fonts, and marker sizes used when drawing bounds on debug artifacts.
# ============================================

COLOR_FUSED: Final[tuple[int, int, int]] = (0, 220, 0)
"""Green: Visual + OCR match (balanced for readability)."""

COLOR_OCR: Final[tuple[int, int, int]] = (0, 230, 220)
"""Yellow/teal: OCR-only detection (balanced for visibility)."""

COLOR_VISUAL: Final[tuple[int, int, int]] = (230, 230, 0)
"""Cyan: Visual-only detection (balanced for visibility)."""

VIZ_MARKER_SIZE: Final[int] = 20
"""Size of the marker used in debug frames (moderate size)."""

VIZ_MARKER_THICKNESS: Final[int] = 2
"""Thickness of the marker lines (visible but not too bold)."""

VIZ_TEXT_SCALE: Final[tuple[float, float, float]] = (0.48, 0.38, 0.33)
"""Font scale for debug label text (main, sub, audit) balanced for readability."""

VIZ_TEXT_THICKNESS: Final[int] = 1
"""Thickness of the font in debug labels."""

VIZ_TEXT_OFFSET: Final[tuple[int, int]] = (16, 25)
"""Pixel offset for placing text relative to the detection marker (X offset, base Y offset)."""

OUTER_THICKNESS: Final[int] = VIZ_MARKER_THICKNESS * 2
"""Outer rectangle thickness for bounding boxes (double the inner thickness)."""

INNER_THICKNESS: Final[int] = VIZ_MARKER_THICKNESS
"""Inner rectangle thickness for bounding boxes (main color border)."""
