"""Provide entry points for the TJM automation pipeline.

Expose runnable automation workflows including VLM-based, CV-based,
and hybrid perception strategies.
"""

from core import set_high_dpi_awareness
from notepad_task import NotepadTask
from strategies import (
    CVStrategy,
    HybridCVFirstStrategy,
    HybridVLMFirstStrategy,
    LaunchStrategy,
    VLMStrategy,
)


def _run(strategy: LaunchStrategy) -> None:
    """Initialize environment and execute the automation pipeline."""
    set_high_dpi_awareness()
    NotepadTask(strategy).run()


def run_vlm() -> None:
    """Execute the automation pipeline using VLM-based grounding."""
    _run(VLMStrategy())


def run_cv() -> None:
    """Execute the automation pipeline using CV-based grounding."""
    _run(CVStrategy())


def run_hybrid_cv_first() -> None:
    """Execute the pipeline with CV primary and VLM fallback."""
    _run(HybridCVFirstStrategy())


def run_hybrid_vlm_first() -> None:
    """Execute the pipeline with VLM primary and CV fallback."""
    _run(HybridVLMFirstStrategy())
