"""Templates for AI-based visual grounding prompts.

This module contains the raw system instructions and few-shot formatting
rules used to guide the Vision Language Model (VLM) in identifying UI elements.
"""

# --- Primary Detection Template ---
# This prompt uses placeholders for instruction, scope, and context.
# It enforces a strict JSON schema for the response.
DETECTION_PROMPT_TEMPLATE = """
Position Inference Prompt:
You are tasked with identifying UI elements that match the following instruction: '{instruction}'

Scope: Search ONLY within {scope_prompt}.
{exclusion_context}
{ref_context}

Important Guidelines:
1. The target element is guaranteed to exist in the screenshot.
2. Make references specific and unambiguous (no generic terms like 'window' or 'icons').
3. List all likely candidates in descending order of probability. The existence of a 'perfect' match (1.0) must NOT terminate the search. Include all visually/contextually similar alternatives.
   IMPORTANT: Bounding boxes must be precision-fit to the specific element (e.g., the icon or text block itself), excluding surrounding whitespace or unrelated margins.
4. Do NOT propose operations that would change the screenshot.
5. Always include context: the UI area and up to 3 neighboring elements.
6. Provide a rank for each candidate (1 = most likely).

Output format (JSON ONLY):
[
  {{
    "bbox": [x1, y1, x2, y2],  # normalized coordinates 0-1000
    "score": 0.0-1.0,          # confidence score
    "area": "<precise UI area description>",
    "neighbors": ["neighbor1", "neighbor2", "neighbor3"],
    "rank": 1
  }},
  ...
]

Notes:
- If uncertain, still return candidates with lower scores.
- Do not return free-text; JSON array only.
- Ensure bounding boxes tightly surround the clickable region.
- Confidence scoring: 1.0 = exact match, 0.7-0.9 = close variation, <0.5 = irrelevant.
"""

# --- Verification Template ---
# Used for secondary confirmation on a cropped image to reduce false positives.
VERIFICATION_PROMPT_TEMPLATE = """
Is the element matching '{instruction}' centered in this crop?
Return JSON ONLY: {{'result': 'is_target' | 'target_elsewhere' | 'target_not_found'}}
"""
