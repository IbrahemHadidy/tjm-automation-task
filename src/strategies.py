"""Provide grounding and UI automation strategies for application launching.

Define the contract for launching applications using different perception
engines (VLM or CV). Include visual artifact logging and hardware
input safety locks to prevent user interference during critical actions.
"""

from __future__ import annotations

import ctypes
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw

from config import ICON_PATH, OPENCV_TEXT_QUERY, TESS_PATH, VLM_INSTRUCTION
from cv_strategy.engine import CVGroundingEngine
from vlm_strategy.engine import AiGroundingEngine
from vlm_strategy.utils import AiImageUtils

if TYPE_CHECKING:
    from cv2.typing import MatLike


# =========================================================
# HELPERS
# =========================================================


def _get_active_notepad_hwnd() -> int | None:
    """Locate the currently visible Notepad window on the desktop.

    Returns:
        The window handle (HWND) if found and visible; None otherwise.

    """
    wins = [
        w
        for w in gw.getWindowsWithTitle(OPENCV_TEXT_QUERY)
        if w.visible and not w.isMinimized
    ]
    return wins[0]._hWnd if wins else None  # noqa: SLF001


def _verify_and_launch(
    x: int,
    y: int,
    score: float,
    source: str,
    min_score: float = 0.5,
) -> int | None:
    """Execute a double-click on a candidate location with a hardware input lock.

    Args:
        x: Target screen X coordinate.
        y: Target screen Y coordinate.
        score: Confidence score from the perception engine (0.0 to 1.0).
        source: Name of the calling strategy for logging/telemetry.
        min_score: Threshold below which the launch attempt is aborted.

    Returns:
        The HWND of the successfully launched application; None if the
        application failed to appear or the score was too low.

    """
    if score < min_score:
        return None

    try:
        # Hardware Lock to prevent User Interference during the critical click
        ctypes.windll.user32.BlockInput(True)  # noqa: FBT003
        print(f"[INFO] Launch via {source} at ({x}, {y}) [Score: {score:.2f}]")
        pyautogui.doubleClick(x, y)
        time.sleep(0.2)
    finally:
        ctypes.windll.user32.BlockInput(False)  # noqa: FBT003

    # Poll for the OS to initialize the window
    for _ in range(6):
        hwnd = _get_active_notepad_hwnd()
        if hwnd:
            print(f"[SUCCESS] {OPENCV_TEXT_QUERY} confirmed via {source}. HWND: {hwnd}")
            return hwnd
        time.sleep(0.5)

    print(f"[WARN] {source} target at ({x}, {y}) failed to open application.")
    return None


# =========================================================
# STRATEGIES
# =========================================================


class LaunchStrategy(ABC):
    """Base class for all UI grounding perception strategies.

    This interface defines the contract for discovering UI elements and
    executing a hardware-validated launch sequence.
    """

    @abstractmethod
    def launch(self) -> int | None:
        """Execute the grounding and launch sequence.

        Returns:
            The Win32 window handle (HWND) of the launched application.

        """

    @abstractmethod
    def get_debug_frame(self) -> MatLike | None:
        """Retrieve the visual perception map from the last launch attempt.

        Returns:
            An OpenCV image showing the perception result, or None if no
            attempt has been made.

        """


class VLMStrategy(LaunchStrategy):
    """Ground applications using Multi-modal VLM reasoning.

    Attributes:
        engine: The AI grounding engine for coordinate resolution.
        last_perception_viz: The BGR image buffer of the last inference result.

    """

    def __init__(self) -> None:
        """Initialize the VLM strategy with a specific debug directory."""
        self.engine = AiGroundingEngine()
        self.last_perception_viz: MatLike | None = None

    def launch(self) -> int | None:
        """Resolve coordinates via VLM and attempt application launch."""
        try:
            results = self.engine.resolve_coordinates(
                instruction=VLM_INSTRUCTION,
                target_window="Desktop",
            )
            if not results:
                return None

            # Sort by highest score, then by rank
            results.sort(key=lambda n: (-n.get("score", 0.0), n.get("rank", 1)))

            # Prepare debug visualization
            screenshot = pyautogui.screenshot()
            viz_pil = AiImageUtils.draw_debug_results(screenshot, results)
            self.last_perception_viz = cv2.cvtColor(
                np.array(viz_pil),
                cv2.COLOR_RGB2BGR,
            )

            for n in results:
                hwnd = _verify_and_launch(
                    n["coords"][0],
                    n["coords"][1],
                    n.get("score", 0.0),
                    "AI-Vision",
                )
                if hwnd:
                    return hwnd

        except Exception as e:
            print(f"[VLM Strategy] Critical Error: {e}")
            return None

        return None

    def get_debug_frame(self) -> MatLike | None:
        """Retrieve the last AI perception frame."""
        return self.last_perception_viz


class CVStrategy(LaunchStrategy):
    """Ground applications using Template Matching and OCR.

    Attributes:
        engine: The Computer Vision engine for template and text matching.

    """

    def __init__(self) -> None:
        """Initialize the CV engine with centralized Tesseract path."""
        self.engine = CVGroundingEngine(tesseract_path=TESS_PATH)

    def launch(self) -> int | None:
        """Locate elements via icon/text matching and attempt launch."""
        screenshot_img = pyautogui.screenshot()
        try:
            results = self.engine.locate_elements(
                screenshot=screenshot_img,
                icon_image=ICON_PATH,
                text_query=OPENCV_TEXT_QUERY,
            )
            results.sort(key=lambda c: c.score, reverse=True)

            for n in results:
                hwnd = _verify_and_launch(n.x, n.y, n.score, "CV")
                if hwnd:
                    return hwnd

        except Exception as e:
            print(f"[CV Strategy] Critical Error: {e}")
            return None

        return None

    def get_debug_frame(self) -> MatLike | None:
        """Retrieve the last CV grounding debug frame."""
        return self.engine.last_debug_frame


class HybridCVFirstStrategy(LaunchStrategy):
    """Attempt grounding via CV first, falling back to VLM on failure.

    Attributes:
        cv: Instance of the CV-based perception strategy.
        vlm: Instance of the VLM-based perception strategy.

    """

    def __init__(self) -> None:
        """Initialize both CV and VLM engines."""
        self.cv = CVStrategy()
        self.vlm = VLMStrategy()
        self._last_used_strategy: LaunchStrategy | None = None

    def launch(self) -> int | None:
        """Execute CV launch and fallback to VLM if unsuccessful."""
        self._last_used_strategy = self.cv
        hwnd = self.cv.launch()
        if hwnd:
            return hwnd

        print("[HYBRID] CV failed. Triggering VLM fallback...")
        self._last_used_strategy = self.vlm
        return self.vlm.launch()

    def get_debug_frame(self) -> MatLike | None:
        """Retrieve the debug frame from the most recently active engine."""
        return (
            self._last_used_strategy.get_debug_frame()
            if self._last_used_strategy
            else None
        )


class HybridVLMFirstStrategy(LaunchStrategy):
    """Attempt grounding via VLM first, falling back to CV on failure.

    Attributes:
        vlm: Instance of the VLM-based perception strategy.
        cv: Instance of the CV-based perception strategy.

    """

    def __init__(self) -> None:
        """Initialize both VLM and CV engines."""
        self.vlm = VLMStrategy()
        self.cv = CVStrategy()
        self._last_used_strategy: LaunchStrategy | None = None

    def launch(self) -> int | None:
        """Execute VLM launch and fallback to CV if unsuccessful."""
        self._last_used_strategy = self.vlm
        hwnd = self.vlm.launch()
        if hwnd:
            return hwnd

        print("[HYBRID] VLM failed. Triggering CV fallback...")
        self._last_used_strategy = self.cv
        return self.cv.launch()

    def get_debug_frame(self) -> MatLike | None:
        """Retrieve the debug frame from the most recently active engine."""
        return (
            self._last_used_strategy.get_debug_frame()
            if self._last_used_strategy
            else None
        )
