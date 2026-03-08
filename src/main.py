"""Provide entry points for the TJM automation pipeline.

Expose runnable automation workflows including LLM-based, CV-based,
and hybrid perception strategies.
"""

from notepad_task import NotepadTask
from strategies import (
    CVStrategy,
    HybridCVFirstStrategy,
    HybridLLMFirstStrategy,
    LLMStrategy,
)


def run_llm() -> None:
    """Execute the automation pipeline using LLM-based grounding."""
    NotepadTask(LLMStrategy()).run()


def run_cv() -> None:
    """Execute the automation pipeline using CV-based grounding."""
    NotepadTask(CVStrategy()).run()


def run_hybrid_cv_first() -> None:
    """Execute the pipeline with CV primary and LLM fallback."""
    NotepadTask(HybridCVFirstStrategy()).run()


def run_hybrid_llm_first() -> None:
    """Execute the pipeline with LLM primary and CV fallback."""
    NotepadTask(HybridLLMFirstStrategy()).run()
