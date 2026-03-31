"""Templates for AI-based visual grounding prompts.

This module contains the raw system instructions and few-shot formatting
rules used to guide the Vision Language Model (VLM) in identifying UI elements.
"""

# --- Primary Detection Template ---
# This prompt uses placeholders for instruction, scope, and context.
# It enforces a strict JSON schema for the response.
DETECTION_PROMPT_TEMPLATE = """
Role: You are an exhaustive visual scanner for a UI automation system.
Task: Locate ALL plausible candidate UI elements that match or relate to the instruction: '{instruction}'

Scope: Search ONLY within {scope_prompt}.
{exclusion_context}

Critical Directives for High Recall:
1. EXHAUSTIVE SEARCH: Do not stop at the first good match. You must mentally scan the entire specified scope from top-left to bottom-right.
2. PLAUSIBILITY OVER PERFECTION: Extract the exact target AND all visually or semantically similar elements (e.g., repeated icons, identical buttons, similar text, or partial matches).
3. MANDATORY MULTIPLICITY: Finding only 1 element is usually a failure of this scan. Extract every single distinct instance. If there are 5 similar icons, return 5 candidates.
4. Provide context for each: Note the precise UI area and up to 3 immediate neighboring elements.
5. Rank them logically: 1 = most exact match, higher numbers = plausible alternatives.

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
"""

# --- Verification Template ---
# Used for secondary confirmation on a cropped image to reduce false positives.
VERIFICATION_PROMPT_TEMPLATE = """
Is the element matching '{instruction}' centered in this crop?
Return JSON ONLY: {{'result': 'is_target' | 'target_elsewhere' | 'target_not_found'}}
"""
