"""LLM Solution: AI-driven visual grounding and UI element detection.

This package provides an orchestration layer for using Vision Language Models
(VLMs) to identify, locate, and verify UI elements on screen via natural
language instructions.
"""

from llm_solution.client import AiClient
from llm_solution.engine import AiGroundingEngine
from llm_solution.models import AIDetection, UIElementNode
from llm_solution.utils import AiImageUtils

__all__ = [
    "AIDetection",
    "AiClient",
    "AiGroundingEngine",
    "AiImageUtils",
    "UIElementNode",
]
