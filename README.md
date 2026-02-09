# AI-Powered Vision Grounding & Notepad Automation

This repository features a cutting-edge **AI LLM Reasoning Engine** designed for semantic, context-aware interaction. While an OpenCV-based module is included for benchmarking, the **LLM-based architecture is the primary solution** for production.

The LLM engine resolves the fundamental flaws of traditional computer vision, specifically the inaccurate scoring logic and the inability of template matching to handle non-default icon sizes.

---

## 🚀 Primary Execution (AI Agent)

This project uses `uv` for lightning-fast dependency management. The LLM commands provide the most stable, production-ready experience.

| Command | Action |
| :--- | :--- |
| **`uv sync`** | Install all dependencies and synchronize the virtual environment. |
| **`uv run start-llm`** | **Launch the primary AI Reasoning GUI.** |
| **`uv run notepad-llm`** | **Execute the automated Notepad task via LLM reasoning.** |
| `uv run start-opencv` | *Optional:* Launch the legacy OpenCV diagnostic lab. |
| `uv run notepad-opencv` | *Optional:* Execute the automated Notepad task via legacy OpenCV automation. |

---

## 📂 Repository Structure (`src/`)

The architecture prioritizes high-level AI reasoning, using shared services only for basic OS-level tasks.

* **`llm_solution/`**: **The Core Engine.** Contains the vision-language model logic and the stable agent interface.  
* **`screenshot_service.py`**: A shared utility for window management and workspace recovery.  
* **`opencv_solution/`**: *Legacy/Optional.* Contains heuristic-based attempts (OCR, templates) that serve as a baseline for the LLM's superior performance.

---

## 🤖 The Primary Solution: LLM Reasoning Engine

The LLM engine is the "brain" of this project. It is the **recommended standard** because it overcomes the mathematical and visual limitations of the legacy engine.

> **Note:** The LLM uses the **Gemini 3 Flash Preview API**. Its performance depends on internet speed, token limits, and potential server overloading. Real-time responsiveness may vary under heavy load.

### 1. Agent Control Center (`gui.py`)

![LLM Grounding Screenshot](screenshots/llm_ss.png)

* **Semantic Intelligence**: Correctly distinguishes between "Notepad" and "Notepad++." Traditional OCR (Tesseract) often returns false positives when similar strings are present.  
* **Resolution Agnostic**: Unlike template matching, the LLM handles non-default desktop icon sizes and DPI scaling without needing constant parameter adjustment.  
* **Stable API Usage**: While fast, using the API may experience latency spikes depending on network speed and server load. Token consumption is tracked per request.

### 2. Intelligent Notepad Automation (`notepad_automation.py`)

* **AI-Located Entry**: Uses the Grounding Engine to find the "Notepad Shortcut" dynamically, ensuring the script works regardless of where the icon is placed on the desktop.  
* **Workspace Recovery**: Automatically captures the state of visible windows before execution and restores them afterward, providing a seamless "non-destructive" automation experience.  
* **Data-to-Disk Pipeline**: Fetches real-time data from external APIs and uses "bomb-proof" keyboard automation to handle save dialogs and file overwrites across various OS versions.

---

## 🔬 Legacy Supplement: OpenCV Grounding (Optional)

The `DesktopGroundingEngine` (OpenCV) is provided strictly for **diagnostic comparison**. It utilizes a heuristic fusion approach (CIELAB, ORB, Tesseract) with refined OCR scoring and template/icon sizing for more accurate desktop element detection.  
It also **omits infinite values in template matching scores**, which previously broke ranking results.

![OpenCV Grounding Screenshot](screenshots/opencv_ss.png)

---

## ⚖️ LLM vs OpenCV: Performance & Semantic Comparison

| Aspect | **LLM Grounding (Primary)** | **OpenCV Grounding (Optional)** |
| :--- | :--- | :--- |
| **Flexibility** | Can handle multiple screens, varying icon sizes, and dynamic layouts without extra configuration. | Works well for fixed templates and simple layouts; additional targets require new templates and tuning. |
| **Template Dependency** | Operates directly on semantic understanding of the screen; template images are **optional**. | Heavy reliance on template images and fixed heuristics. |
| **Performance** | Slightly slower due to reasoning overhead and API calls. Dependent on internet speed, token limits, and potential server overloading. | Faster for small, controlled tasks; can be customized via configuration (passes, ROI size, confidence thresholds, thread count). |
| **Accuracy** | High; adaptive scoring considers context, partial matches, and UI structure. | Moderate; improved with ROI adjustments, OCR scoring, and template tuning, but still brittle in non-default layouts. |
| **Scalability & Maintenance** | Easy to extend to new apps, screens, or workflows without extra setup. | Adding new targets requires new template images and manual tuning. |

**Summary:**  

* **Use LLM** for production workflows, semantic understanding, and long-term stability.
* **Use OpenCV** for diagnostics, quick experiments, or highly controlled automation where performance and configurability matter more than adaptability.  

---

## 🛠 Prerequisites

1. **Python 3.14** and **UV** package manager.  
2. **Environment Variables**: A `.env` file with your GEMINI API key is **required** to power the Reasoning Engine.  
3. **Tesseract (Optional)**: Only required if you intend to benchmark the legacy OpenCV tools.
