"""Provide grounding and UI automation strategies for application launching.

Define the contract for launching applications using different perception
engines (VLM or CV). Include visual artifact logging and hardware
input safety locks to prevent user interference during critical actions.
"""

from __future__ import annotations

import ctypes
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
from vlm_solution.engine import AiGroundingEngine
from vlm_solution.utils import AiImageUtils

from cv_solution.engine import CVGroundingEngine

if TYPE_CHECKING:
    from cv2.typing import MatLike

# Configuration Constants
ICON_PATH = Path("notepad_icon.png")
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
VLM_INSTRUCTION = "Notepad shortcut"
OPENCV_TEXT_QUERY = "Notepad"

# =========================================================
# HELPERS
# =========================================================


def _get_active_notepad() -> gw.Win32Window | None:
    """Locate the currently visible Notepad window on the desktop.

    Returns:
        The window object if found and visible, otherwise None.

    """
    wins = [
        w for w in gw.getWindowsWithTitle("Notepad") if w.visible and not w.isMinimized
    ]
    return wins[0] if wins else None


def _verify_and_launch(
    x: int,
    y: int,
    score: float,
    source: str,
    min_score: float = 0.5,
) -> bool:
    """Execute a double-click on a candidate location using a hardware input lock.

    Args:
        x, y: Target screen coordinates.
        score: Confidence score from the perception engine.
        source: Name of the strategy (e.g., 'CV') for logging.
        min_score: Threshold below which the attempt is aborted.

    Returns:
        True if Notepad is detected after the interaction.

    """
    if score < min_score:
        return False

    try:
        # Safety Lock: Prevent user mouse/keyboard movement from breaking the click
        # Note: Requires Administrator privileges
        ctypes.windll.user32.BlockInput(True)  # noqa: FBT003

        print(
            f"[INFO] Attempting launch via {source} at ({x}, {y}) [Score: {score:.2f}]",
        )
        pyautogui.doubleClick(x, y)
        time.sleep(0.2)
    finally:
        # Always release the hardware lock
        ctypes.windll.user32.BlockInput(False)  # noqa: FBT003

    # Polling: Wait for the OS to initialize the window
    for _ in range(6):
        if _get_active_notepad():
            print(f"[SUCCESS] Notepad confirmed via {source}.")
            return True
        time.sleep(0.5)

    print(f"[WARN] {source} target at ({x}, {y}) failed to open application.")
    return False


# =========================================================
# STRATEGIES
# =========================================================


class LaunchStrategy(ABC):
    """Act as the base class for all UI grounding perception strategies."""

    @abstractmethod
    def launch(self) -> bool:
        """Execute the grounding and launch sequence.

        Returns:
            True if the target application was successfully launched.

        """

    @abstractmethod
    def get_debug_frame(self) -> MatLike | None:
        """Retrieve the visual perception map from the last launch attempt."""


class VLMStrategy(LaunchStrategy):
    """Ground applications using Multi-modal VLM reasoning."""

    def __init__(self, debug_dir: str = "logs/ai_debug") -> None:
        """Initialize the AI grounding engine with a specific debug path."""
        self.engine = AiGroundingEngine(debug_dir=debug_dir)
        self.last_perception_viz: MatLike | None = None

    def launch(self) -> bool:
        """Resolve coordinates using AI vision and attempt the launch sequence.

        Workflow:
        1. Resolve target coordinates via the AI engine.
        2. Generate a 'Magenta Auditor' artifact with bounding boxes.
        3. Iterate through candidates in order of confidence.
        """
        try:
            # 1. Request AI Predictions
            results = self.engine.resolve_coordinates(
                instruction=VLM_INSTRUCTION,
                target_window="Desktop",
            )

            if not results:
                print("[VLM Strategy] AI found no valid targets.")
                return False

            # 2. Sort by confidence scores
            results.sort(key=lambda n: (-n.get("score", 0.0), n.get("rank", 1)))

            # 3. Create a Perception Map (Magenta Audit Artifact)
            screenshot = pyautogui.screenshot()
            viz_pil = AiImageUtils.draw_debug_results(screenshot, results)

            # Convert to CV format (BGR) for the monitoring system
            self.last_perception_viz = cv2.cvtColor(
                np.array(viz_pil),
                cv2.COLOR_RGB2BGR,
            )

            # 4. Execute attempts sequentially until one succeeds
            return any(
                _verify_and_launch(
                    n["coords"][0],
                    n["coords"][1],
                    n.get("score", 0.0),
                    "AI-Vision",
                )
                for n in results
            )

        except Exception as e:
            print(f"[VLM Strategy] Critical Error: {e}")
            return False

    def get_debug_frame(self) -> MatLike | None:
        """Retrieve the AI-generated perception map with coordinate markers."""
        return self.last_perception_viz


class CVStrategy(LaunchStrategy):
    """Ground applications using Template Matching and OCR."""

    def __init__(self) -> None:
        """Initialize the CV engine with the system Tesseract path."""
        self.engine = CVGroundingEngine(tesseract_path=TESS_PATH)

    def launch(self) -> bool:
        """Locate elements via icon matching or OCR and attempt launch.

        Use in-memory screenshots to perform multi-pass detection.
        """
        screenshot_img = pyautogui.screenshot()

        try:
            # 1. Locate elements using the CV engine
            results = self.engine.locate_elements(
                screenshot=screenshot_img,
                icon_image=ICON_PATH,
                text_query=OPENCV_TEXT_QUERY,
            )

            # 2. Prioritize high-confidence matches
            results.sort(key=lambda c: c.score, reverse=True)

            # 3. Iterate through matches
            return any(_verify_and_launch(n.x, n.y, n.score, "CV") for n in results)

        except Exception as e:
            print(f"[CV Strategy] Critical Error: {e}")
            return False

    def get_debug_frame(self) -> MatLike | None:
        """Retrieve the CV debug frame containing detection overlays."""
        return self.engine.last_debug_frame


class HybridCVFirstStrategy(LaunchStrategy):
    """Attempt grounding via CV first, falling back to VLM on failure."""

    def __init__(self) -> None:
        """Initialize both CV and VLM perception engines."""
        self.cv = CVStrategy()
        self.vlm = VLMStrategy()
        self._last_used_strategy: LaunchStrategy | None = None

    def launch(self) -> bool:
        """Execute CV launch and fallback to VLM if unsuccessful."""
        self._last_used_strategy = self.cv
        if self.cv.launch():
            return True

        print("[HYBRID] CV failed. Triggering VLM fallback...")
        self._last_used_strategy = self.vlm
        return self.vlm.launch()

    def get_debug_frame(self) -> MatLike | None:
        """Retrieve the debug frame from the most recently attempted strategy."""
        return (
            self._last_used_strategy.get_debug_frame()
            if self._last_used_strategy
            else None
        )


class HybridVLMFirstStrategy(LaunchStrategy):
    """Attempt grounding via VLM first, falling back to CV on failure."""

    def __init__(self) -> None:
        """Initialize both VLM and CV perception engines."""
        self.vlm = VLMStrategy()
        self.cv = CVStrategy()
        self._last_used_strategy: LaunchStrategy | None = None

    def launch(self) -> bool:
        """Execute VLM launch and fallback to CV if unsuccessful."""
        self._last_used_strategy = self.vlm
        if self.vlm.launch():
            return True

        print("[HYBRID] VLM failed. Triggering CV fallback...")
        self._last_used_strategy = self.cv
        return self.cv.launch()

    def get_debug_frame(self) -> MatLike | None:
        """Retrieve the debug frame from the most recently attempted strategy."""
        return (
            self._last_used_strategy.get_debug_frame()
            if self._last_used_strategy
            else None
        )
