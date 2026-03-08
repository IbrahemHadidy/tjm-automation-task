"""Centralize configuration constants for computer vision and OCR engines.

This module defines the thresholds, scaling factors, and scoring weights used
across the detection pipeline. All 'FACTOR' constants are relative to the
target template size (e.g., 0.5 = 50% of icon width/height).
"""

# ============================================
# Tesseract Engine Settings
# ============================================

# Engine Mode (OEM) 3: Default, based on what is available (Legacy + LSTM).
OCR_ENGINE_MODE = 3

# Page Segmentation Modes (PSM)
# 7: Treat the image as a single text line (Best for isolated labels).
# 3: Fully automatic page segmentation (Best for full-screen search).
PSM_SINGLE_LINE = 7
PSM_AUTO = 3


# ============================================
# OCR Thresholds & Scoring
# ============================================

# Minimum Tesseract confidence (0-100) and string length to accept a token.
OCR_MIN_CONFIDENCE = 10
OCR_MIN_TOKEN_LENGTH = 2

# Radius in pixels to merge nearly overlapping OCR tokens into a single candidate.
TOKEN_DEDUP_RADIUS_PX = 15

# Minimum similarity score (0.0 - 1.0) for fuzzy matching and ROI recovery.
OCR_RECOVERY_THRESHOLD = 0.5
OCR_FUZZY_MATCH_THRESHOLD = 0.6

# Scoring weights for string similarity logic.
OCR_EXACT_MATCH_SCORE = 1.0  # Perfect string equality.
OCR_SUBSTRING_BASE_SCORE = 0.5  # Starting point if query is inside a longer string.
OCR_SUBSTRING_MATCH_WEIGHT = 0.35  # Multiplier for the length ratio of the match.
OCR_SIMILARITY_WEIGHT = 0.5  # Weight for Levenshtein distance in fuzzy mode.

# Penalty applied to recovery matches (0.05) to ensure Global Search results
# take precedence during Non-Maximum Suppression (NMS).
OCR_RECOVERY_PENALTY = 0.05

# Internal Mode IDs for the image preprocessing pipeline.
OCR_MODE_OTSU = 1
OCR_MODE_INVERTED_OTSU = 2
OCR_MODE_CUBIC_UPSCALE = 5
OCR_MODE_GRAY = 8
OCR_MODE_TOPHAT_UPSCALE_A = 11
OCR_MODE_TOPHAT_UPSCALE_B = 12

# Modes executed during the initial full-screen 'Global' search pass.
OCR_GLOBAL_SEARCH_MODES = [
    OCR_MODE_OTSU,
    OCR_MODE_INVERTED_OTSU,
    OCR_MODE_CUBIC_UPSCALE,
    OCR_MODE_GRAY,
    OCR_MODE_TOPHAT_UPSCALE_A,
    OCR_MODE_TOPHAT_UPSCALE_B,
]


# ============================================
# Image Preprocessing & Scaling
# ============================================

# Standard bit-depth maximum for 8-bit grayscale images.
MAX_8BIT_VALUE = 255

# Interpolation methods (Mapping to cv2.INTER_* constants).
INTERPOLATION_UP = 3  # cv2.INTER_CUBIC (Best for upscaling).
INTERPOLATION_LOCAL = 1  # cv2.INTER_LINEAR (Standard resizing).

# Color Space Conversions (Mapping to cv2.COLOR_* constants).
BGR_TO_GRAY = 6  # cv2.COLOR_BGR2GRAY
BGR_TO_LAB = 44  # cv2.COLOR_BGR2Lab

# Multipliers for image resolution enhancement before OCR.
OCR_GLOBAL_UPSCALE_FACTOR = 2.5
OCR_LOCAL_UPSCALE_FACTOR = 2.0

# Kernel size for morphological Top-hat filtering to isolate text from noisy backgrounds.
OCR_MORPH_KERNEL_SIZE = (9, 9)


# ============================================
# Visual Engine Technical Settings
# ============================================

# Template Matching Method (5 = cv2.TM_CCOEFF_NORMED).
TPL_MATCH_METHOD = 5

# Distance metric for feature matching (6 = cv2.NORM_HAMMING for ORB).
ORB_NORM_TYPE = 6


# ============================================
# Geometry & Aspect Ratio Validation
# ============================================

# The scoring formula is: final_score = original_score * (BASE + (BONUS * deviation))
# GEOM_BASE_SCORE_WEIGHT: Trust floor for any detected shape.
GEOM_BASE_SCORE_WEIGHT = 0.8
# GEOM_RATIO_BONUS_WEIGHT: Extra confidence for perfect aspect ratio matches.
GEOM_RATIO_BONUS_WEIGHT = 0.2


# ============================================
# Template Matching Thresholds
# All values range from 0.0 (permissive) to 1.0 (strict).
# ============================================

TPL_COLOR_THRESHOLD = 0.7
TPL_LAB_THRESHOLD = 0.7
TPL_EDGE_THRESHOLD = 0.4  # Lower due to high sensitivity of Canny edges.
TPL_GRAY_THRESHOLD = 0.65
TPL_MULTISCALE_THRESHOLD = 0.65

# Canny Edge detection parameters for the EDGE method.
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Scale factors for multiscale template matching.
MULTISCALE_FACTORS = (0.8, 1.25)

# Limit raw hits per method to prevent NMS performance degradation.
MAX_TEMPLATE_HITS = 200

# Deduplication Radii (Multipliers of template size).
NMS_RADIUS_FACTOR = 0.6  # Intra-method overlap removal.
FUSION_DISTANCE_FACTOR = 1.5  # Max distance to pair a visual hit with an OCR label.
FINAL_DEDUP_RADIUS_FACTOR = 0.7  # Final cross-method cleanup.
MIN_DEDUPE_RADIUS_FACTOR = 0.8  # Safety floor for deduplication.

# Bonus added to the score when both Visual AND OCR confirm the same element.
FUSION_SCORE_BONUS = 0.1


# ============================================
# Recovery Search Geometry
# Relative to the anchor icon's center.
# ============================================

RECOVERY_HORIZONTAL_PAD_FACTOR = 0.4  # Width extension for text search.
RECOVERY_VERTICAL_EXTEND_FACTOR = 1.6  # Height extension (labels are usually below).
RECOVERY_VERTICAL_OFFSET_PX = 5  # Slight downward shift for the search ROI.


# ============================================
# Desktop Detection & Performance
# ============================================

TASKBAR_HEIGHT_PX = 60
ICON_MIN_WIDTH = 30
ICON_MAX_WIDTH = 150
DEFAULT_ICON_SIZE = 64

# Limit the number of sub-regions processed per frame to maintain real-time speeds.
RECOVERY_QUEUE_LIMIT = 12

# Feature-based matching (ORB) parameters.
ORB_MIN_MATCHES = 15
ORB_SAMPLE_POINTS = 20
ORB_MAX_FEATURES = 1000
ORB_DEFAULT_SCORE = 0.9


# ============================================
# Debug Visualization Settings (BGR)
# ============================================

COLOR_FUSED = (0, 255, 0)  # Green
COLOR_OCR = (0, 255, 255)  # Yellow
COLOR_VISUAL = (255, 255, 0)  # Cyan

VIZ_MARKER_SIZE = 25
VIZ_MARKER_THICKNESS = 2
VIZ_TEXT_SCALE = 0.4
VIZ_TEXT_THICKNESS = 1
VIZ_TEXT_OFFSET = (18, 15)
