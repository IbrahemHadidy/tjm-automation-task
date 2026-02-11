"""Configuration constants for computer vision and OCR engine.

This file centralizes thresholds for template matching, text recognition (OCR),
and spatial geometry logic used to detect and deduplicate desktop UI elements.
"""

# ============================================
# Template Matching Thresholds
# ============================================

# Minimum confidence for template matches
TPL_COLOR_THRESHOLD = 0.7
TPL_LAB_THRESHOLD = 0.7
TPL_EDGE_THRESHOLD = 0.4
TPL_GRAY_THRESHOLD = 0.65

# Edge detection (Canny)
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Multiscale matching factors
MULTISCALE_FACTORS = (0.8, 1.25)

# Max template hits kept before NMS
MAX_TEMPLATE_HITS = 200

# Distance-based NMS factors
NMS_RADIUS_FACTOR = 0.6
FUSION_DISTANCE_FACTOR = 1.5
FINAL_DEDUP_RADIUS_FACTOR = 0.7

# Score boost applied when template and OCR hits are fused
FUSION_SCORE_BONUS = 0.1


# ============================================
# OCR Thresholds
# ============================================

OCR_MIN_CONFIDENCE = 10
OCR_MIN_TOKEN_LENGTH = 2

# Pixel radius used to merge nearly overlapping OCR tokens
TOKEN_DEDUP_RADIUS_PX = 15

# Text similarity acceptance
OCR_RECOVERY_THRESHOLD = 0.5

# OCR similarity scoring weights
OCR_EXACT_MATCH_SCORE = 1.0
OCR_SUBSTRING_BASE_SCORE = 0.5
OCR_SIMILARITY_WEIGHT = 0.5


# ============================================
# Recovery Search Geometry
# ============================================

RECOVERY_HORIZONTAL_PAD_FACTOR = 0.4
RECOVERY_VERTICAL_EXTEND_FACTOR = 1.6


# ============================================
# Desktop Detection
# ============================================

TASKBAR_HEIGHT_PX = 60

ICON_MIN_WIDTH = 30
ICON_MAX_WIDTH = 150
DEFAULT_ICON_SIZE = 64


# ============================================
# Performance Limits
# ============================================

RECOVERY_QUEUE_LIMIT = 12
ORB_MIN_MATCHES = 15
ORB_SAMPLE_POINTS = 20
