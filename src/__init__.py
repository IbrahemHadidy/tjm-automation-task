"""TJM Automation Framework.

A production-grade automation pipeline utilizing dual-mode perception
(LLM & Computer Vision) to orchestrate desktop applications.

Core Components:
    - NotepadTask: The FSM controller for the automation lifecycle.
    - LaunchStrategy: Abstract base for perception engines.
    - ScreenshotService: Isolated UI capture utility.
    - RunMonitor: Telemetry and artifact management.

Usage:
    >>> from src import run_llm
    >>> run_llm()
"""

from .main import run_hybrid_llm_first, run_hybrid_opencv_first, run_llm, run_opencv

__all__ = [
    "run_hybrid_llm_first",
    "run_hybrid_opencv_first",
    "run_llm",
    "run_opencv",
]
