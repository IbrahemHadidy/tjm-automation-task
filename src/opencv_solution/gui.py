"""Provide a graphical interface for the Desktop Grounding Engine.

This module implements a PyQt6-based laboratory for testing computer vision
and OCR detection strategies on desktop screenshots.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from opencv_solution.grounding_engine import DesktopGroundingEngine

if TYPE_CHECKING:
    import numpy as np

# --- STYLING CONSTANTS ---
BG_COLOR, SURFACE_COLOR, ACCENT_COLOR = "#0F0F0F", "#1A1A1A", "#00E5FF"
SUCCESS_COLOR, WARNING_COLOR, ERROR_COLOR = "#00FF88", "#FFD600", "#FF3D00"

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
    QLineEdit, QDoubleSpinBox, QSpinBox {{
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

    def set_frame(self, frame: np.ndarray | None) -> None:
        """Update the internal high-resolution frame used for magnification."""
        self.full_res_frame = frame.copy() if frame is not None else None
        self.update()

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: ANN401, N802
        """Trigger a repaint when the mouse moves to update the zoom overlay."""
        self.update()
        super().mouseMoveEvent(event)

    def paintEvent(self, event: Any) -> None:  # noqa: ANN401, N802
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

    def leaveEvent(self, event: Any) -> None:  # noqa: ANN401, N802
        """Clear the zoom overlay when the mouse leaves the viewport."""
        self.update()
        super().leaveEvent(event)


class Worker(QThread):
    """Handle the execution of the grounding engine in a background thread."""

    finished = pyqtSignal(list, object)
    log_signal = pyqtSignal(str, str)
    frame_signal = pyqtSignal(object)
    progress_signal = pyqtSignal(int)

    def __init__(
        self,
        engine: DesktopGroundingEngine,
        params: list,
        config: dict,
    ) -> None:
        """Initialize the worker thread with the engine instance and search parameters."""
        super().__init__()
        self.engine, self.params, self.config = engine, params, config

    def run(self) -> None:
        """Execute the engine search logic and bridge signals back to the main UI."""
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
                    hasattr(self.engine, "debug_frame")
                    and self.engine.debug_frame is not None
                ):
                    self.frame_signal.emit(self.engine.debug_frame)

            results = self.engine.locate_elements(
                *self.params,
                config=self.config,
                callback=callback_bridge,
            )

            if self.isInterruptionRequested():
                self.log_signal.emit("PROCESS TERMINATED BY USER", "WARNING")
                self.finished.emit([], None)
            else:
                self.progress_signal.emit(100)
                self.finished.emit(results, getattr(self.engine, "debug_frame", None))

        except Exception as e:
            self.log_signal.emit(f"RUNTIME ERROR: {e!s}", "ERROR")
            self.finished.emit([], None)


class GroundingLab(QMainWindow):
    """Orchestrate the main application window and UI interactions."""

    def __init__(self) -> None:
        """Initialize the main window and setup UI components."""
        super().__init__()
        self.setWindowTitle("VISION GROUNDING ENGINE v0.2")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(GLOBAL_STYLE)
        self.img_path: str | None = None
        self.icon_path: str | None = None
        self.last_results: list = []
        self.worker: Worker | None = None
        self.init_ui()

    def init_ui(self) -> None:  # noqa: PLR0915
        """Construct the layout, sidebars, and viewport widgets."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        sidebar = QVBoxLayout()

        # --- 1. INPUT GROUP ---
        input_group = QGroupBox("Target Inputs")
        input_f = QFormLayout(input_group)
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter text label to find...")
        self.btn_ss = QPushButton("SELECT SCREENSHOT")
        self.btn_icon = QPushButton("SELECT ICON")
        input_f.addRow("QUERY", self.query_input)
        input_f.addRow(self.btn_ss)
        input_f.addRow(self.btn_icon)
        sidebar.addWidget(input_group)

        # --- 2a. TEMPLATE MATCHING PASSES ---
        template_group = QGroupBox("Template Matching Passes")
        template_v = QVBoxLayout(template_group)
        self.chk_color = QCheckBox("Color Match (BGR)")
        self.chk_lab = QCheckBox("CIELAB Match (Lighting)")
        self.chk_edge = QCheckBox("Edge/Canny Match")
        self.chk_gray = QCheckBox("Grayscale Match")
        self.chk_orb = QCheckBox("ORB (Rotation Invariant)")
        self.chk_multiscale = QCheckBox("Multi-Scale Sweep")

        for chk in [
            self.chk_color,
            self.chk_lab,
            self.chk_edge,
            self.chk_gray,
            self.chk_orb,
            self.chk_multiscale,
        ]:
            chk.setChecked(True)
            template_v.addWidget(chk)
        sidebar.addWidget(template_group)

        # --- 2b. OCR PASSES ---
        ocr_group = QGroupBox("OCR Passes")
        ocr_v = QVBoxLayout(ocr_group)
        self.chk_ocr = QCheckBox("Enable OCR Engine")
        self.chk_adaptive = QCheckBox("Adaptive Threshold")
        self.chk_sharpen = QCheckBox("OCR Sharpening")
        self.chk_upscale = QCheckBox("OCR 2x Upscale")
        self.chk_iso = QCheckBox("RGB Isolation")
        self.chk_fusion = QCheckBox("Heuristic Fusion")

        for chk in [
            self.chk_ocr,
            self.chk_adaptive,
            self.chk_sharpen,
            self.chk_upscale,
            self.chk_iso,
            self.chk_fusion,
        ]:
            chk.setChecked(True)
            ocr_v.addWidget(chk)
        sidebar.addWidget(ocr_group)

        # --- CONNECT INPUT EVENTS ---
        self.btn_ss.clicked.connect(lambda: self.get_file("ss"))
        self.btn_icon.clicked.connect(
            lambda: [self.get_file("icon"), self.update_toggle_states()],
        )
        self.query_input.textChanged.connect(lambda _: self.update_toggle_states())

        # --- 3. SYSTEM CONFIG ---
        sys_group = QGroupBox("Engine Config")
        sys_f = QFormLayout(sys_group)
        self.tess_path = QLineEdit(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 1.0)
        self.threshold.setValue(0.80)
        self.num_cores = QSpinBox()
        self.num_cores.setRange(1, 32)
        self.num_cores.setValue(8)
        sys_f.addRow("TESS PATH", self.tess_path)
        sys_f.addRow("CONFIDENCE", self.threshold)
        sys_f.addRow("THREADS", self.num_cores)
        sidebar.addWidget(sys_group)

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
        sidebar.addWidget(action_group)

        self.btn_copy = QPushButton("COPY BEST COORDINATES")
        self.btn_dump = QPushButton("DUMP SYSTEM LOGS")
        self.btn_reset = QPushButton("RESET STATE")
        sidebar.addWidget(self.btn_reset)
        self.btn_reset.clicked.connect(self.reset_state)
        sidebar.addWidget(self.btn_copy)
        sidebar.addWidget(self.btn_dump)
        sidebar.addStretch()
        main_layout.addLayout(sidebar, 1)

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
        main_layout.addLayout(viewport, 4)

        # --- CONNECT ACTION BUTTONS ---
        self.btn_run.clicked.connect(self.run_engine)
        self.btn_cancel.clicked.connect(self.cancel_engine)
        self.btn_copy.clicked.connect(self.copy_best)
        self.btn_dump.clicked.connect(self.dump_logs)

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
            <span style='color:{color}; font-weight:bold;'>[{lvl}]</span><br>
            <pre
                style='
                    font-family: "Consolas", "Monaco", monospace;
                    margin: 0;
                    white-space: pre;
                '
            >
                {m}
            </pre>
        </div>
        """
        self.console.append(styled_message)
        vbar = self.console.verticalScrollBar()
        if vbar:
            vbar.setValue(vbar.maximum())

    def get_file(self, mode: str) -> None:
        """Open a file dialog to select input screenshots or icons."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if path:
            if mode == "ss":
                self.img_path = path
                self.btn_ss.setText(f"✓ {Path(path).name}")
                frame = cv2.imread(path)
                if frame is not None:
                    self.update_view(frame)
            else:
                self.icon_path = path
                self.btn_icon.setText(f"✓ {Path(path).name}")

    def update_toggle_states(self) -> None:
        """Update UI elements based on whether icon or text input is present."""
        icon_selected = bool(self.icon_path)
        text_entered = bool(self.query_input.text().strip())

        # --- TEMPLATE PASS CHECKBOXES ---
        template_checkboxes = [
            self.chk_color,
            self.chk_lab,
            self.chk_edge,
            self.chk_gray,
            self.chk_orb,
            self.chk_multiscale,
        ]
        for chk in template_checkboxes:
            chk.setEnabled(icon_selected)
            chk.setToolTip("" if icon_selected else "Select an icon first")

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
        """Reset the entire UI to its initial state."""
        # Clear file paths
        self.img_path = None
        self.icon_path = None

        # Reset buttons
        self.btn_ss.setText("SELECT SCREENSHOT")
        self.btn_icon.setText("SELECT ICON")

        # Clear text input
        self.query_input.clear()

        # Reset checkboxes
        for chk in [
            self.chk_color,
            self.chk_lab,
            self.chk_edge,
            self.chk_gray,
            self.chk_orb,
            self.chk_multiscale,
            self.chk_ocr,
            self.chk_adaptive,
            self.chk_sharpen,
            self.chk_upscale,
            self.chk_iso,
            self.chk_fusion,
        ]:
            chk.setChecked(True)

        # Reset engine config
        self.threshold.setValue(0.80)
        self.num_cores.setValue(8)

        # Reset progress and cancel button
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_run.setObjectName("action_btn_idle")
        self.btn_run.setText("START DIAGNOSTICS")
        self.btn_cancel.hide()

        # Clear viewport
        self.display.set_frame(None)
        self.display.clear()

        # Clear console
        self.console.clear()

        # Clear last results
        self.last_results = []

        # Update toggle states (to correctly disable checkboxes)
        self.update_toggle_states()

    def run_engine(self) -> None:
        """Start the grounding engine diagnostics in a background thread."""
        if not self.img_path:
            return self.log("Reference screenshot required!", "ERROR")

        self.btn_run.setEnabled(False)
        self.btn_run.setObjectName("action_btn_loading")
        self.btn_run.setText("ANALYZING...")
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.show()
        self.progress_bar.setValue(0)

        engine = DesktopGroundingEngine(self.tess_path.text())
        config = {
            t: getattr(self, f"chk_{t.split('_')[1]}").isChecked()
            for t in [
                "use_color",
                "use_lab",
                "use_edge",
                "use_gray",
                "use_orb",
                "use_multiscale",
                "use_adaptive",
                "use_ocr",
            ]
        }
        config["num_cores"] = self.num_cores.value()

        params = [
            Path(self.img_path),
            Path(self.icon_path) if self.icon_path else None,
            self.query_input.text(),
            self.threshold.value(),
            11,
            2.0,
        ]

        self.worker = Worker(engine, params, config)
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
        """Convert a BGR frame to RGB and update the display label."""
        if frame is None:
            return
        self.display.set_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        self.display.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ),
        )

    def on_finished(self, results: list, frame: cv2.Mat | None) -> None:
        """Restore UI state and store detection results after thread completion."""
        self.btn_run.setEnabled(True)
        self.btn_run.setObjectName("action_btn_idle")
        self.btn_run.setText("START DIAGNOSTICS")
        self.btn_cancel.hide()
        self.last_results = results
        if frame is not None:
            self.update_view(frame)

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

    def resizeEvent(self, event: Any) -> None:  # noqa: ANN401, N802
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
