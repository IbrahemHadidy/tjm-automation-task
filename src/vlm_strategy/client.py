"""Client for interacting with Google GenAI models for visual grounding."""

import json
import re
from typing import TYPE_CHECKING

from google import genai

if TYPE_CHECKING:
    from PIL import Image

    from vlm_strategy import AIDetection


class AiClient:
    """Handles direct communication and JSON parsing for the GenAI API.

    This client encapsulates the Google GenAI SDK to send multi-modal prompts
    and extract structured grounding data from conversational responses.
    """

    def __init__(self, api_key: str, model_id: str) -> None:
        """Initialize the Google GenAI client.

        Args:
            api_key: The Google API Key for authentication.
            model_id: The specific Gemini model ID (e.g., 'gemini-3.1-pro-preview').

        """
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id

    def generate_detection(self, prompt: str, image: Image.Image) -> list[AIDetection]:
        """Send a prompt and image to the model and returns parsed detections.

        Args:
            prompt: The instruction text describing what to find.
            image: A PIL Image object of the current screen.

        Returns:
            A list of AIDetection dictionaries found by the model.
            Returns an empty list if the API fails or no JSON is found.

        """
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt, image],
            )
            return self._parse_json_list(response.text or "")
        except Exception as e:
            print(f"[ERROR] AiClient API failure: {e}")
            return []

    def generate_verification(self, prompt: str, image: Image.Image) -> str:
        """Send a verification prompt and image to the model.

        Args:
            prompt: Verification instruction.
            image: Cropped region to verify.

        Returns:
            Raw model response text.

        """
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt, image],
            )
        except Exception as e:
            print(f"[ERROR] AiClient verification failure: {e}")
            return "error"
        return response.text or "unknown"

    def _parse_json_list(self, text: str) -> list[AIDetection]:
        """Extract and parses a JSON array from the raw AI string response.

        Models often wrap JSON in markdown or conversational text. This
        method uses regex to isolate the primary JSON array.

        Args:
            text: The raw string response from the GenAI model.

        Returns:
            A list of parsed AIDetection objects.

        """
        try:
            # Matches the first square bracket block found in the response
            # re.DOTALL ensures the dot matches newlines within the JSON block
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError, AttributeError:
            print("[WARN] AiClient failed to parse JSON from response.")
            return []
        return []
