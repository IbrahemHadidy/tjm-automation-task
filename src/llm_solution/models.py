"""Data models and type definitions for the LLM grounding solution."""

from typing import TypedDict


class AIDetection(TypedDict):
    """Raw detection result returned by the vision model.

    Attributes:
        bbox: A list of four integers [x1, y1, x2, y2] normalized to a
            0-1000 scale representing the bounding box.
        score: Confidence score from 0.0 to 1.0.
        area: Semantic description of the UI region (e.g., 'top-right toolbar').
        neighbors: List of text/descriptions of elements near the target.
        rank: The priority assigned by the model (1 = best match).

    """

    bbox: list[int]
    score: float
    area: str
    neighbors: list[str]
    rank: int


class UIElementNode(TypedDict):
    """Represent a localized UI element identified and processed by the engine.

    Attributes:
        coords: [x, y] center pixel coordinates on the actual screen.
        score: Confidence score (0.0 to 1.0).
        area: Description of the UI region.
        neighbors: List of nearby UI element descriptions.
        rank: Priority ranking.
        size: Optional [width, height] in pixels.

    """

    coords: list[int]
    score: float
    area: str
    neighbors: list[str]
    rank: int
    size: list[int] | None
