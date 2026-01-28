# Vision Grounding Engine & Notepad Automation

An advanced computer vision toolkit designed to bridge the gap between high-level intent and low-level GUI interaction. This repository provides a dual-engine architecture, offering both a high-speed **OpenCV Grounding Engine** for deterministic precision and an **AI LLM Reasoning Engine** for semantic, context-aware desktop interaction.

---

## 🚀 Execution Commands

This project uses `uv` for lightning-fast dependency management and script execution.

| Command | Action |
| :--- | :--- |
| **`uv sync`** | Install all dependencies and synchronize the virtual environment. |
| **`uv run start-opencv`** | Open the OpenCV-based PyQt6 Diagnostic Lab. |
| **`uv run notepad-opencv`** | Execute the Notepad automation task using the OpenCV engine. |
| **`uv run start-llm`** | Open the LLM-based Reasoning GUI. |
| **`uv run notepad-llm`** | Execute the Notepad automation task using the LLM engine. |

---

## 📂 Repository Structure (`src/`)

The repository is organized to separate low-level vision logic from high-level AI reasoning while sharing core utility services.

* **`screenshot_service.py`**: A shared service used by both engines to capture the desktop or specific application windows. It includes **Workspace Recovery** logic to minimize windows for a clean scan and restore them post-execution.

### 📁 `opencv_solution/` & `llm_solution/`

Both solution folders contain a unified set of components:

1. **`grounding_engine.py`**: The "brain" of the solution. The OpenCV version uses heuristic fusion (OCR, templates, edges), while the LLM version uses vision-language models to locate coordinates.
2. **`gui.py`**: The PyQt6 graphical interface tailored to that specific engine's parameters and debug views.
3. **`notepad_automation.py`**: The end-to-end automation script that utilizes its respective engine and `pyautogui` to perform the requested task.

---

## 🚀 Key Components

### 1. Grounding Lab (`gui.py`)

Implement a high-performance **PyQt6** diagnostic laboratory to stress-test detection strategies and fine-tune engine parameters in real-time.

* **Magnifying Viewport**: Hover over the screenshot to inspect pixel-perfect details at 3.0x zoom using the custom magnifying loupe.
* **Multi-Pass Tuning**: Toggle between BGR color matching, CIELAB lighting-invariant matching, ORB rotation-invariant features, and OCR engine passes.
* **Threaded Execution**: Ensure all processing happens in a background worker (`QThread`) to keep the UI responsive.
* **Diagnostic Logs**: Provide real-time console feedback with the ability to export system logs for debugging.

### 2. Notepad Automation (`notepad_automation.py`)

Execute a production-ready automation script that demonstrates the power of the Grounding Engine.

* **Visual Launching**: Find the Notepad icon or text label on a cluttered desktop to launch the application.
* **Data Integration**: Fetch live data from a REST API and handle the "Save As" flow across different Windows versions.

---

## 🛠 Prerequisites

* **Python 3.10+** and **UV** package manager.
* **Tesseract OCR**: Install the engine and ensure the path is correctly set in the GUI (Default: `C:\Program Files\Tesseract-OCR\tesseract.exe`).
* **Environment Variables**: Create a `.env` file in the root directory. Look for **`example.env`** in the repository for the correct variables (e.g., `API_URL`).

---

## 🖥 Usage

### OpenCV Grounding Lab (Diagnostics)

Utilize this interface to benchmark and fine-tune computer vision parameters.

![OpenCV Grounding Screenshot](screenshots/opencv_ss.png)

> **Diagnostic Marker Guide:**
>
> * **Green Squares**: Successful **Fused Matches** where multiple detection strategies agree.
> * **Cyan 'X'**: Candidates found via **Template Matching** only.
> * **Yellow 'X'**: Candidates found via the **OCR Engine** only.
> * **Scoring**: Numerical confidence values are displayed next to each marker to indicate the strength of the match.

1. **Target Inputs**: Load a reference screenshot and define your search via a Text Query or Icon image.
2. **Processing Passes**: Select active detection algorithms (e.g., CIELAB, ORB, OCR).
3. **Engine Config**: Set the **Confidence Threshold** and the number of CPU threads.
4. **Execution**: Click **START DIAGNOSTICS** to initiate the search.
5. **Visual Debugging**: Hover over the viewport to use the **Magnifying Loupe**; verify if detection marks (bounding boxes/points) rendered on the frame align perfectly with the target.
6. **Output**: Click **COPY BEST COORDINATES** to export the top-ranked [X, Y] location or **DUMP SYSTEM LOGS** to save the diagnostic history.

### LLM Reasoning GUI (Agent Control)

Leverage this interface for high-level task execution powered by large language models.

![LLM Grounding Screenshot](screenshots/llm_ss.png)

> **Diagnostic Marker Guide:**
>
> * **Green '+' Sign**: Marks the precise coordinates determined by the AI's spatial reasoning.
> * **Scoring**: Displays the probability/confidence score for each coordinate, indicating how likely it is to be the correct target.

1. **Instruction**: Type a natural language command (e.g., "Find the Notepad icon and click it").
2. **Reasoning Pipeline**: Monitor the console to see the step-by-step logic used to map your instruction to the screen.
3. **Visual Validation**: Verify that the detected **Green '+'** marks align with your target elements before clicking **RUN**.

4. **Target Scope**: Select the operational area, such as the "Entire Desktop."
5. **Instruction**: Type your natural language command (e.g., "Find the Notepad icon and click it").
6. **Visual Anchor**: Click **LOAD** to provide a reference image if the task requires specific visual grounding.
7. **Reasoning Pipeline**: Monitor this section to see the step-by-step logic used to map instructions to coordinates.
8. **Mapped Coordinates**: Review the generated **[X, Y]** locations and their associated probability scores.
9. **Execution**: Click **RUN** to initiate the automated sequence.
10. **Visual Validation**: Verify that the detected coordinate marks on the screen align with your target elements; this serves as the final debug check.

---

## ⚖️ Pros & Cons

### OpenCV Grounding Engine

* **Pros**:
  * **Speed**: Near-instant detection compared to LLM-based vision.
  * **Precision**: Provides exact pixel coordinates ($X, Y$) and bounding boxes.
  * **Privacy**: All processing happens locally; no desktop screenshots are sent to the cloud.
  * **Reliability**: Deterministic results—if the template matches, it will find it every time.
* **Cons**:
  * **Rigidity**: Can struggle with dynamic UI changes (e.g., hover effects or dark mode shifts).
  * **Sensitivity**: Requires high-quality reference icons for the best results.

### LLM Reasoning Layer

* **Pros**:
  * **Context Aware**: Understands "semantic" commands even if the text isn't a literal match.
  * **Flexibility**: Handles variations in UI layout and can "reason" through multi-step tasks.
* **Cons**:
  * **Latency**: Takes significantly longer to process and "think."
  * **Cost**: Usually requires an external API key and consumes tokens.

---

## ⚙️ OpenCV Grounding Engine Configuration

The core detection engine uses a heuristic fusion approach to rank candidates. Customize the behavior via the `config` dictionary (accessible via the "Processing Passes" sidebar):

| Parameter | Description |
| :--- | :--- |
| **use_ocr** | Enable Tesseract OCR for text-based element localization. |
| **use_color** | Apply BGR color distribution filtering to matches. |
| **use_multiscale** | Perform template matching at multiple resolutions (0.5x to 1.5x). |
| **use_lab** | Use CIELAB color space for better accuracy under varying light. |
| **use_orb** | Utilize ORB feature descriptors for rotation and scale invariance. |
| **use_edge** | Match elements based on structural outlines and shapes (Canny edge detection). |
| **use_adaptive** | Use local pixel intensity to handle varying lighting for better segmentation. |
| **use_iso** | Enable RGB Isolation to prioritize specific color channels during the search. |
| **num_cores** | Parallelize the search across multiple CPU cores (Default: 8). |
