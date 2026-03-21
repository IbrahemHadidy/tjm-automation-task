"""Provide a graphical interface for the Desktop Grounding Engine.

This module implements a PySide6-based laboratory for testing computer vision
and OCR detection strategies on desktop screenshots.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pygetwindow as gw
from PySide6.QtCore import QEvent, Qt, QThread, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from cv_strategy.engine import CVGroundingEngine, GroundingConfig
from screenshot_service import ScreenshotService

if TYPE_CHECKING:
    from cv2.typing import MatLike

    from cv_strategy.models import Candidate

# --- STYLING CONSTANTS ---
BG_COLOR = "#0F0F0F"
SURFACE_COLOR = "#1A1A1A"
ACCENT_COLOR = "#00E5FF"
SUCCESS_COLOR = "#00FF88"
WARNING_COLOR = "#FFD600"
ERROR_COLOR = "#FF3D00"

GLOBAL_STYLE = f"""
    QMainWindow {{
        background-color: {BG_COLOR};
    }}
    QWidget {{
        background-color: {BG_COLOR};
        color: #E0E0E0;
        font-family: 'Segoe UI', 'Consolas';
        font-size: 13px;
    }}
    QGroupBox {{
        border: 1px solid #333;
        border-radius: 6px;
        margin-top: 20px;
        font-weight: bold;
        color: {ACCENT_COLOR};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        top: 10px;
    }}
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
        background-color: {SURFACE_COLOR};
        border: 1px solid #444;
        padding: 6px;
        border-radius: 3px;
        color: white;
    }}
    QLineEdit:focus {{
        border-color: {ACCENT_COLOR};
    }}
    QPushButton {{
        background-color: {SURFACE_COLOR};
        border: 1px solid #444;
        border-radius: 4px;
        padding: 10px;
        font-weight: bold;
        text-transform: uppercase;
    }}
    QPushButton:hover {{
        background-color: #252525;
        border-color: {ACCENT_COLOR};
    }}
    QPushButton#action_btn_idle {{
        background-color: {ACCENT_COLOR};
        color: black; border: none;
    }}
    QPushButton#action_btn_loading {{
        background-color: {WARNING_COLOR};
        color: black;
        border: none;
    }}
    QProgressBar {{
        border: 1px solid #333;
        border-radius: 4px;
        text-align: center;
        background-color: #050505;
        height: 10px;
    }}
    QProgressBar::chunk {{
        background-color: {ACCENT_COLOR};
    }}
    QTextEdit {{
        background-color: #050505;
        border: 1px solid #222;
        font-family: 'Consolas';
        font-size: 11px;
    }}
    QToolTip {{
        background-color: #222222;
        color: {ACCENT_COLOR};
        border: 1px solid {ACCENT_COLOR};
        padding: 5px;
        border-radius: 3px;
        font-family: 'Consolas';
    }}
"""


class ZoomableLabel(QLabel):
    """Implement a label with a magnifying glass effect on hover."""

    def __init__(self, text: str = "SYSTEM IDLE") -> None:
        """Initialize the zoomable viewport with default magnification settings."""
        super().__init__(text)
        self.setMouseTracking(True)
        self.full_res_frame = None
        self.zoom_factor = 3.0
        self.zoom_size = 300

    def set_frame(self, frame: MatLike | None) -> None:
        """Update the internal high-resolution frame used for magnification."""
        self.full_res_frame = frame.copy() if frame is not None else None
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Trigger a repaint when the mouse moves to update the zoom overlay."""
        self.update()
        super().mouseMoveEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the scaled image and the magnification loupe if a frame is loaded."""
        super().paintEvent(event)
        if (
            self.full_res_frame is None
            or not self.underMouse()
            or self.pixmap() is None
        ):
            return

        pixmap = self.pixmap()
        pw, ph = pixmap.width(), pixmap.height()
        dx, dy = (self.width() - pw) // 2, (self.height() - ph) // 2
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        ix, iy = cursor_pos.x() - dx, cursor_pos.y() - dy

        if 0 <= ix <= pw and 0 <= iy <= ph:
            oh, ow = self.full_res_frame.shape[:2]
            rx, ry = int((ix / pw) * ow), int((iy / ph) * oh)
            cw = int(self.zoom_size / self.zoom_factor)
            x1, y1 = max(0, rx - cw // 2), max(0, ry - cw // 2)
            crop = self.full_res_frame[y1 : y1 + cw, x1 : x1 + cw]

            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                h, w, _ = crop_rgb.shape
                qimg = QImage(
                    crop_rgb.data.tobytes(),
                    w,
                    h,
                    3 * w,
                    QImage.Format.Format_RGB888,
                )
                zoom_p = QPixmap.fromImage(qimg).scaled(
                    self.zoom_size,
                    self.zoom_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
                p = QPainter(self)
                tx = (
                    cursor_pos.x() + 30
                    if cursor_pos.x() + self.zoom_size + 30 < self.width()
                    else cursor_pos.x() - self.zoom_size - 30
                )
                ty = (
                    cursor_pos.y() + 30
                    if cursor_pos.y() + self.zoom_size + 30 < self.height()
                    else cursor_pos.y() - self.zoom_size - 30
                )
                p.drawPixmap(tx, ty, zoom_p)
                p.setPen(QPen(QColor(ACCENT_COLOR), 2))
                p.drawRect(tx, ty, self.zoom_size, self.zoom_size)
                p.setPen(QColor(255, 255, 255))
                p.drawText(tx + 5, ty + 15, f"REL: {rx}, {ry}")
                p.end()

    def leaveEvent(self, event: QEvent) -> None:
        """Clear the zoom overlay when the mouse leaves the viewport."""
        self.update()
        super().leaveEvent(event)


class Worker(QThread):
    """Handle the execution of the grounding engine in a background thread."""

    finished = Signal(list, object)
    log_signal = Signal(str, str)
    frame_signal = Signal(object)
    progress_signal = Signal(int)

    def __init__(
        self,
        engine: CVGroundingEngine,
        config: GroundingConfig,
        window_title: str,
        icon_path: Path | None,
        text_query: str,
        threshold: float,
    ) -> None:
        """Initialize the worker thread with the engine instance and search parameters."""
        super().__init__()

        self.engine = engine
        self.config = config

        self.window_title = window_title
        self.icon_path = icon_path
        self.text_query = text_query
        self.threshold = threshold

    def run(self) -> None:
        """Execute engine logic using PIL images from the ScreenshotService."""
        try:
            self.engine.should_abort = self.isInterruptionRequested

            def callback_bridge(
                msg: str,
                lvl: str = "INFO",
                progress: int | None = None,
            ) -> None:
                if self.isInterruptionRequested():
                    return
                self.log_signal.emit(msg, lvl)
                if progress is not None:
                    self.progress_signal.emit(progress)
                if (
                    hasattr(self.engine, "last_debug_frame")
                    and self.engine.last_debug_frame is not None
                ):
                    self.frame_signal.emit(self.engine.last_debug_frame)

            shot_service = ScreenshotService()

            # Handle PIL Objects
            if self.window_title == "Desktop":
                self.log_signal.emit("CAPTURING Desktop (PIL Mode)", "INFO")
                active_image = shot_service.capture_desktop()
            else:
                self.log_signal.emit(f"ISOLATING WINDOW: {self.window_title}", "INFO")
                active_image, _ = shot_service.capture_app_window(
                    window_title=self.window_title,
                )

            # Pass the PIL image directly to the engine
            results = self.engine.locate_elements(
                screenshot=active_image,
                icon_image=self.icon_path,
                text_query=self.text_query,
                config=self.config,
                callback=callback_bridge,
            )

            if self.isInterruptionRequested():
                self.log_signal.emit("PROCESS TERMINATED BY USER", "WARNING")
                self.finished.emit([], None)
            else:
                self.progress_signal.emit(100)
                raw_cv2 = cv2.cvtColor(np.array(active_image), cv2.COLOR_RGB2BGR)
                self.finished.emit(results, raw_cv2)

        except Exception as e:
            self.log_signal.emit(f"RUNTIME ERROR: {e!s}", "ERROR")
            self.finished.emit([], None)


class GroundingLab(QMainWindow):
    """Orchestrate the main application window and UI interactions."""

    def __init__(self) -> None:
        """Initialize the main window and setup UI components."""
        super().__init__()
        self.setWindowTitle("VISION GROUNDING LAB")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(GLOBAL_STYLE)
        self.icon_path: str | None = None
        self.last_results: list = []
        self.worker: Worker | None = None
        self.init_ui()

    def init_ui(self) -> None:
        """Construct the layout, sidebars, and viewport widgets."""
        # Create reference for defaults
        d = GroundingConfig()
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(10)
        sidebar_widget.setMinimumWidth(360)
        sidebar_widget.setMaximumWidth(420)

        # --- 1. INPUT GROUP ---
        input_group = QGroupBox("Target Inputs")
        input_f = QFormLayout(input_group)
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter text label to find...")
        self.query_input.textChanged.connect(self.update_toggle_states)

        self.btn_icon = QPushButton("SELECT ICON")
        self.btn_icon.clicked.connect(self.select_icon)

        self.window_selector = QComboBox()
        self.btn_refresh = QPushButton("⟳")
        self.btn_refresh.setFixedWidth(40)
        self.btn_refresh.clicked.connect(self.refresh_windows)

        window_container = QWidget()
        window_container_layout = QHBoxLayout(window_container)
        window_container_layout.setContentsMargins(0, 0, 0, 0)
        window_container_layout.setSpacing(5)
        window_container_layout.addWidget(self.window_selector)
        window_container_layout.addWidget(self.btn_refresh)

        input_f.addRow("QUERY", self.query_input)
        input_f.addRow(self.btn_icon)
        input_f.addRow("TARGET", window_container)
        sidebar_layout.addWidget(input_group)

        # --- 2a. TEMPLATE MATCHING PASSES ---
        template_group = QGroupBox("Template Matching Passes")
        template_v = QVBoxLayout(template_group)

        self.chk_color = QCheckBox("Color Match (BGR)")
        self.chk_lab = QCheckBox("CIELAB Match (Lighting)")
        self.chk_edge = QCheckBox("Edge/Canny Match")
        self.chk_gray = QCheckBox("Grayscale Match")
        self.chk_orb = QCheckBox("ORB (Rotation Invariant)")
        self.chk_multiscale = QCheckBox("Multi-Scale Sweep")

        template_map = {
            self.chk_color: (
                "use_color",
                "Matches pixel-perfect BGR values. Best for static icons.",
            ),
            self.chk_lab: (
                "use_lab",
                "Matches based on Perceptual Lightness. Use for UI with shadows/glows.",
            ),
            self.chk_edge: (
                "use_edge",
                "Matches shapes/outlines only. High success for wireframe icons.",
            ),
            self.chk_gray: (
                "use_gray",
                "Ignores color data. Fastest processing speed.",
            ),
            self.chk_orb: (
                "use_orb",
                "Feature-based matching. Works even if icon is rotated or skewed.",
            ),
            self.chk_multiscale: (
                "use_multiscale",
                "Repeats search at different sizes. Use if icon scaling is unknown.",
            ),
        }

        for chk, (attr, tip) in template_map.items():
            chk.setChecked(getattr(d, attr))
            chk.setToolTip(tip)
            template_v.addWidget(chk)
        sidebar_layout.addWidget(template_group)

        # --- 2b. OCR PASSES ---
        ocr_group = QGroupBox("OCR Passes")
        ocr_v = QVBoxLayout(ocr_group)

        self.chk_ocr = QCheckBox("Enable OCR Engine")
        self.chk_adaptive = QCheckBox("Adaptive Threshold")
        self.chk_sharpen = QCheckBox("OCR Sharpening")
        self.chk_upscale = QCheckBox("OCR 2x Upscale")
        self.chk_iso = QCheckBox("RGB Isolation")
        self.chk_fusion = QCheckBox("Heuristic Fusion")

        ocr_map = {
            self.chk_ocr: ("use_ocr", "Global toggle for the Tesseract Engine."),
            self.chk_adaptive: (
                "use_adaptive",
                "Dynamic binarization. Essential for text on complex backgrounds.",
            ),
            self.chk_sharpen: (
                "use_sharpen",
                "Applies Unsharp Masking to improve character legibility.",
            ),
            self.chk_upscale: (
                "use_upscale",
                "Resamples small text to 2x size before OCR to increase accuracy.",
            ),
            self.chk_iso: (
                "use_iso",
                "Isolates specific color channels to eliminate background noise.",
            ),
            self.chk_fusion: (
                "use_fusion",
                "Combines multiple results using fuzzy logic to pick the best match.",
            ),
        }

        for chk, (attr, tip) in ocr_map.items():
            chk.setChecked(getattr(d, attr, True))
            chk.setToolTip(tip)
            ocr_v.addWidget(chk)
        sidebar_layout.addWidget(ocr_group)

        # --- 3. SYSTEM CONFIG ---
        sys_group = QGroupBox("Engine Config")
        sys_f = QFormLayout(sys_group)
        self.tess_path = QLineEdit(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 1.0)
        self.threshold.setValue(0.70)
        self.threshold.setToolTip(
            "Minimum confidence score (0.0 - 1.0) to accept a match.",
        )
        self.num_cores = QSpinBox()
        self.num_cores.setRange(1, 32)
        self.num_cores.setValue(8)
        self.num_cores.setToolTip(
            "Number of CPU threads for parallel template processing.",
        )
        self.psm_input = QSpinBox()
        self.psm_input.setRange(0, 13)
        self.psm_input.setValue(11)
        self.psm_input.setToolTip(
            "Tesseract Page Segmentation Mode. 11 = Sparse text, 3 = Fully automatic.",
        )
        self.min_icon_width = QSpinBox()
        self.min_icon_width.setRange(10, 500)
        self.min_icon_width.setValue(30)
        self.max_icon_width = QSpinBox()
        self.max_icon_width.setRange(50, 1000)
        self.max_icon_width.setValue(150)
        icon_size_layout = QHBoxLayout()
        icon_size_layout.addWidget(QLabel("Min Width"))
        icon_size_layout.addWidget(self.min_icon_width)
        icon_size_layout.addWidget(QLabel("Max Width"))
        icon_size_layout.addWidget(self.max_icon_width)

        sys_f.addRow("TESS PATH", self.tess_path)
        sys_f.addRow("CONFIDENCE", self.threshold)
        sys_f.addRow("THREADS", self.num_cores)
        sys_f.addRow("PSM MODE", self.psm_input)
        sys_f.addRow("ICON SIZE", icon_size_layout)
        sidebar_layout.addWidget(sys_group)

        # --- 4. PROGRESS & ACTIONS ---
        action_group = QGroupBox("Engine Control")
        action_v = QVBoxLayout(action_group)
        self.progress_bar = QProgressBar()
        self.btn_run = QPushButton("START DIAGNOSTICS")
        self.btn_run.setObjectName("action_btn_idle")
        self.btn_cancel = QPushButton("STOP ENGINE")
        self.btn_cancel.setStyleSheet(
            f"background-color: {ERROR_COLOR}; color: white; border: none;",
        )
        self.btn_cancel.hide()
        action_v.addWidget(self.progress_bar)
        action_v.addWidget(self.btn_run)
        action_v.addWidget(self.btn_cancel)
        sidebar_layout.addWidget(action_group)

        self.btn_copy = QPushButton("COPY BEST COORDINATES")
        self.btn_dump = QPushButton("DUMP SYSTEM LOGS")
        self.btn_reset = QPushButton("RESET STATE")
        sidebar_layout.addWidget(self.btn_reset)
        self.btn_reset.clicked.connect(self.reset_state)
        sidebar_layout.addWidget(self.btn_copy)
        sidebar_layout.addWidget(self.btn_dump)
        sidebar_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(sidebar_widget)

        main_layout.addWidget(scroll)
        main_layout.setStretch(0, 1)

        # --- VIEWPORT ---
        viewport = QVBoxLayout()
        self.display = ZoomableLabel()
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display.setStyleSheet(
            "background: #050505; border: 2px solid #1A1A1A; border-radius: 4px;",
        )
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        viewport.addWidget(self.display, 8)
        viewport.addWidget(self.console, 2)
        main_layout.addLayout(viewport)
        main_layout.setStretch(1, 4)
        self.refresh_windows()

        # --- CONNECT ACTION BUTTONS ---
        self.btn_run.clicked.connect(self.run_engine)
        self.btn_cancel.clicked.connect(self.cancel_engine)
        self.btn_copy.clicked.connect(self.copy_best)
        self.btn_dump.clicked.connect(self.dump_logs)

    def select_icon(self) -> None:
        """Open a file dialog to select a template icon image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Template Icon",
            "",
            "Images (*.png *.jpg *.bmp *.jpeg)",
        )
        if path:
            self.icon_path = path
            self.btn_icon.setText(f"ICON: {Path(path).name}")
            self.update_toggle_states()

    def refresh_windows(self) -> None:
        """Populate the window selector with titles of all active desktop windows."""
        self.window_selector.clear()
        self.window_selector.addItem("Desktop")

        titles = [w.title for w in gw.getAllWindows() if w.title.strip()]
        self.window_selector.addItems(sorted(titles))

        # Ensure "Desktop" is selected by default
        self.window_selector.setCurrentIndex(0)
        self.log(f"Detected {len(titles)} active windows. Default: Desktop")

    def log(self, m: str, lvl: str = "INFO") -> None:
        """Append a styled HTML message to the system console."""
        colors = {
            "SUCCESS": SUCCESS_COLOR,
            "ERROR": ERROR_COLOR,
            "INFO": "#888",
            "WARNING": WARNING_COLOR,
            "HEAD": ACCENT_COLOR,
        }
        color = colors.get(lvl, "#bbb")

        styled_message = f"""
        <div style='margin-bottom: 8px;'>
            <span style='color:{color}; font-weight:bold;'>[{lvl}]</span>
            <pre
                style='
                    font-family: "Consolas", "Monaco", monospace;
                    margin: 0;
                    white-space: pre;
                '
            >{m}</pre>
        </div><br>
        """
        self.console.append(styled_message)
        vbar = self.console.verticalScrollBar()
        if vbar:
            vbar.setValue(vbar.maximum())

    def update_toggle_states(self) -> None:
        """Update UI elements based on whether icon or text input is present."""
        icon_selected = bool(self.icon_path)
        text_entered = bool(self.query_input.text().strip())

        # --- TEMPLATE PASS CHECKBOXES ---
        for chk in [
            self.chk_color,
            self.chk_lab,
            self.chk_edge,
            self.chk_gray,
            self.chk_orb,
            self.chk_multiscale,
        ]:
            chk.setEnabled(icon_selected)
            if not icon_selected:
                chk.setToolTip("DISABLED: Select an icon first")

        # --- OCR PASS CHECKBOXES ---
        ocr_checkboxes = [
            self.chk_ocr,
            self.chk_adaptive,
            self.chk_sharpen,
            self.chk_upscale,
            self.chk_iso,
            self.chk_fusion,
        ]
        for chk in ocr_checkboxes:
            chk.setEnabled(text_entered)
            chk.setToolTip("" if text_entered else "Enter text to enable OCR options")

        # --- RUN BUTTON ---
        self.btn_run.setEnabled(icon_selected or text_entered)
        self.btn_run.setToolTip(
            ""
            if icon_selected or text_entered
            else "Provide a template or text to start",
        )

    def reset_state(self) -> None:
        """Reset the entire UI to its initial state using GroundingConfig defaults."""
        # Create a reference for default values
        d = GroundingConfig()

        # Clear file paths
        self.icon_path = None

        # Reset buttons
        self.btn_icon.setText("SELECT ICON")

        # Clear text input
        self.query_input.clear()

        # Reset checkboxes based on GroundingConfig defaults
        self.chk_color.setChecked(d.use_color)
        self.chk_lab.setChecked(d.use_lab)
        self.chk_edge.setChecked(d.use_edge)
        self.chk_gray.setChecked(d.use_gray)
        self.chk_orb.setChecked(d.use_orb)
        self.chk_multiscale.setChecked(d.use_multiscale)
        self.chk_ocr.setChecked(d.use_ocr)
        self.chk_adaptive.setChecked(d.use_adaptive)

        # Note: If these aren't in your dataclass yet, they will use True
        self.chk_sharpen.setChecked(getattr(d, "use_sharpen", True))
        self.chk_upscale.setChecked(getattr(d, "use_upscale", True))
        self.chk_iso.setChecked(getattr(d, "use_iso", True))
        self.chk_fusion.setChecked(getattr(d, "use_fusion", True))

        # Reset engine config from dataclass defaults
        self.threshold.setValue(d.threshold)
        self.psm_input.setValue(d.psm)
        self.num_cores.setValue(d.num_cores)
        self.min_icon_width.setValue(d.min_icon_width)
        self.max_icon_width.setValue(d.max_icon_width)

        # Reset progress and cancel button
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_run.setObjectName("action_btn_idle")
        self.btn_run.setText("START DIAGNOSTICS")
        self.btn_run.setStyleSheet("")  # Clear any loading styles
        self.btn_cancel.hide()

        # Clear viewport
        self.display.set_frame(None)
        self.display.clear()
        self.display.setText("SYSTEM IDLE")

        # Clear console and results
        self.console.clear()
        self.last_results = []

        # Update toggle states (to correctly disable checkboxes)
        self.update_toggle_states()

    def run_engine(self) -> None:
        """Start the grounding engine diagnostics in a background thread."""
        target_window = self.window_selector.currentText()
        if not target_window:
            return self.log("Reference target window required!", "ERROR")

        self.btn_run.setEnabled(False)
        self.btn_run.setObjectName("action_btn_loading")
        self.btn_run.setText("ANALYZING...")
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.show()
        self.progress_bar.setValue(0)

        engine = CVGroundingEngine(self.tess_path.text())
        config = GroundingConfig(
            use_color=self.chk_color.isChecked(),
            use_lab=self.chk_lab.isChecked(),
            use_edge=self.chk_edge.isChecked(),
            use_gray=self.chk_gray.isChecked(),
            use_orb=self.chk_orb.isChecked(),
            use_multiscale=self.chk_multiscale.isChecked(),
            use_ocr=self.chk_ocr.isChecked(),
            use_adaptive=self.chk_adaptive.isChecked(),
            num_cores=self.num_cores.value(),
            threshold=self.threshold.value(),
            psm=self.psm_input.value(),
            min_icon_width=self.min_icon_width.value(),
            max_icon_width=self.max_icon_width.value(),
        )

        self.worker = Worker(
            engine,
            config,
            target_window,
            Path(self.icon_path) if self.icon_path else None,
            self.query_input.text(),
            self.threshold.value(),
        )
        self.worker.log_signal.connect(self.log)
        self.worker.frame_signal.connect(self.update_view)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        return None

    def cancel_engine(self) -> None:
        """Request the worker thread to stop its current detection pass."""
        if self.worker and self.worker.isRunning():
            self.log("ABORT REQUESTED...", "WARNING")
            self.worker.requestInterruption()
            self.btn_cancel.setEnabled(False)
            self.btn_cancel.setText("STOPPING...")

    def update_view(self, frame: np.ndarray | None) -> None:
        """Update display using in-memory frame with smooth scaling."""
        if frame is None:
            return

        self.display.set_frame(frame)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)

        # Scale to fit view while keeping aspect ratio
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.display.setPixmap(pixmap)

    def on_finished(self, results: list[Candidate], frame: np.ndarray | None) -> None:
        """Restore UI state and display the engine's internal debug view instead of manually drawing annotations."""
        self.btn_run.setEnabled(True)
        self.btn_run.setObjectName("action_btn_idle")
        self.btn_run.setText("START DIAGNOSTICS")
        self.btn_cancel.hide()
        self.last_results = results

        # Determine which frame to display as the final view
        final_view = frame

        # If the engine has a specific debug frame (the 'last_debug'), use that.
        # This shows the binary/edge/filtered image the engine actually 'saw'.
        if (
            self.worker
            and hasattr(self.worker.engine, "last_debug_frame")
            and self.worker.engine.last_debug_frame is not None
        ):
            final_view = self.worker.engine.last_debug_frame

        if final_view is not None:
            self.update_view(final_view)

        if results:
            self.log(f"Detection complete. Found {len(results)} candidates.", "SUCCESS")
        else:
            self.log("Detection complete. No matches found.", "WARNING")

    def copy_best(self) -> None:
        """Copy the coordinates of the highest-ranking candidate to the clipboard."""
        if not self.last_results:
            return self.log("No results!", "ERROR")
        best = self.last_results[0]
        x, y = getattr(best, "x", 0), getattr(best, "y", 0)
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(f"{x}, {y}")
            self.log(f"Copied {x}, {y}", "SUCCESS")

        return None

    def dump_logs(self) -> None:
        """Export the console text to a local log file."""
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Logs",
            "vision.log",
            "Logs (*.log)",
        )
        if path_str:
            output_path = Path(path_str)
            with output_path.open("w", encoding="utf-8") as f:
                f.write(self.console.toPlainText())

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle window resizing by scaling the current viewport image."""
        if self.display.full_res_frame is not None:
            self.update_view(self.display.full_res_frame)
        super().resizeEvent(event)


def run() -> None:
    """Execute the application main loop."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = GroundingLab()
    gui.showMaximized()
    sys.exit(app.exec())
