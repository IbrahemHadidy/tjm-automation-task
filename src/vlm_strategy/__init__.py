"""VLM Solution: AI-driven visual grounding and UI element detection.

This package provides an orchestration layer for using Vision Language Models
(VLMs) to identify, locate, and verify UI elements on screen via natural
language instructions.
"""

from vlm_strategy.client import AiClient
from vlm_strategy.engine import AiGroundingEngine
from vlm_strategy.models import AIDetection, UIElementNode
from vlm_strategy.utils import AiImageUtils

__all__ = [
    "AIDetection",
    "AiClient",
    "AiGroundingEngine",
    "AiImageUtils",
    "UIElementNode",
]
