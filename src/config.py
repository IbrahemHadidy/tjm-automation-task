"""Centralized configuration management for the automation framework.

Loads environment variables and defines global constants to prevent
magic strings and scattered settings across the codebase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Project Paths ---
PROJECT_DIR = Path.home() / "Desktop" / "tjm-project"
LOG_DIR = PROJECT_DIR / "logs"
ASSETS_DIR = Path(__file__).parent / "assets"
ICON_PATH = ASSETS_DIR / "notepad_icon.png"

# --- Third-Party Dependencies ---
TESS_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# --- Execution Limits ---
MAX_POSTS = int(os.getenv("MAX_POSTS", "10"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
API_TIMEOUT_SEC = int(os.getenv("API_TIMEOUT_SEC", "10"))
GLOBAL_TIMEOUT_SEC = int(os.getenv("GLOBAL_TIMEOUT_SEC", "900"))

# --- API ---
API_URL = os.getenv("API_URL")
if not API_URL:
    msg = "API_URL missing from environment variables"
    raise ValueError(msg)


# --- Perception Queries ---
# The string expected by OCR/OpenCV when searching for the window title
OPENCV_TEXT_QUERY = os.getenv("OPENCV_TEXT_QUERY", "Notepad")

# The natural language prompt sent to the VLM grounding engine
VLM_INSTRUCTION = os.getenv("VLM_INSTRUCTION", "Notepad shortcut")
