"""Provide specialized detection processors for the grounding pipeline.

Export visual, OCR, and fusion processors used to identify and
reconcile multi-modal UI detection candidates.
"""

from .fusion import FusionProcessor
from .ocr import OCRProcessor
from .visual import VisualProcessor

__all__ = ["FusionProcessor", "OCRProcessor", "VisualProcessor"]
