"""Expose core Computer Vision grounding capabilities.

Provide a unified interface for desktop UI element detection via
template matching and OCR engines.
"""

from .engine import CVGroundingEngine
from .models import GroundingConfig, GroundingResult

__all__ = ["CVGroundingEngine", "GroundingConfig", "GroundingResult"]
