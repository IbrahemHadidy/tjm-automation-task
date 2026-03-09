"""TJM Automation Framework.

A production-grade automation pipeline utilizing dual-mode perception
(VLM & Computer Vision) to orchestrate desktop applications.

Core Components:
    - NotepadTask: The FSM controller for the automation lifecycle.
    - LaunchStrategy: Abstract base for perception engines.
    - ScreenshotService: Isolated UI capture utility.
    - RunMonitor: Telemetry and artifact management.

Usage:
    >>> from src import run_vlm
    >>> run_vlm()
"""

from .main import run_hybrid_opencv_first, run_hybrid_vlm_first, run_opencv, run_vlm

__all__ = [
    "run_hybrid_opencv_first",
    "run_hybrid_vlm_first",
    "run_opencv",
    "run_vlm",
]
