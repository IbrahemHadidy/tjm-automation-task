"""Provide entry points for the TJM automation pipeline.

Expose runnable automation workflows including VLM-based, CV-based,
and hybrid perception strategies.
"""

from notepad_task import NotepadTask
from strategies import (
    CVStrategy,
    HybridCVFirstStrategy,
    HybridVLMFirstStrategy,
    VLMStrategy,
)


def run_vlm() -> None:
    """Execute the automation pipeline using VLM-based grounding."""
    NotepadTask(VLMStrategy()).run()


def run_cv() -> None:
    """Execute the automation pipeline using CV-based grounding."""
    NotepadTask(CVStrategy()).run()


def run_hybrid_cv_first() -> None:
    """Execute the pipeline with CV primary and VLM fallback."""
    NotepadTask(HybridCVFirstStrategy()).run()


def run_hybrid_vlm_first() -> None:
    """Execute the pipeline with VLM primary and CV fallback."""
    NotepadTask(HybridVLMFirstStrategy()).run()
