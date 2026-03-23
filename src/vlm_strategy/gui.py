"""Provide a diagnostic laboratory interface for AI grounding and vision tasks."""

import os
import sys

import pygetwindow as gw
from PIL import ImageQt
from PySide6.QtCore import QEvent, Qt, QThread, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core import set_high_dpi_awareness
from vlm_strategy.engine import AiGroundingEngine, UIElementNode

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"
os.environ["QT_API"] = "pyqt6"

# UI Constants
BG_DARK = "#0B0E14"
SURFACE = "#151921"
ACCENT = "#00F0FF"
SUCCESS = "#39FF14"
ERROR = "#FF3131"


class AIWorker(QThread):
    """Resolve AI coordinates in a background thread."""

    log_signal = Signal(str, str)
    result_signal = Signal(list)
    finished_signal = Signal()

    def __init__(
        self,
        engine: AiGroundingEngine,
        instruction: str,
        target_window: str,
        ref_path: str | None,
        *,
        verify_after_action: bool,
    ) -> None:
        """Initialize the worker thread with task parameters."""
        super().__init__()
        self.engine = engine
        self.instruction = instruction
        self.target_window = target_window
        self.ref_path = ref_path
        self.verify_after_action = verify_after_action
        self._is_cancelled = False

    def cancel(self) -> None:
        """Stop the worker processing."""
        self._is_cancelled = True

    def run(self) -> None:
        """Execute the AI resolution pipeline."""
        try:
            if self._is_cancelled:
                return

            def engine_logger(msg: str) -> None:
                """Emit engine logs to the UI via signals."""
                if not self._is_cancelled:
                    level = "INFO"
                    if "[SUCCESS]" in msg:
                        level = "SUCCESS"
                    elif "[ERROR]" in msg:
                        level = "ERROR"

                    clean_msg = (
                        msg.replace("[INFO]", "")
                        .replace("[SUCCESS]", "")
                        .replace("[ERROR]", "")
                        .strip()
                    )
                    self.log_signal.emit(clean_msg, level)

            results = self.engine.resolve_coordinates(
                instruction=self.instruction,
                target_window=self.target_window,
                reference_image_path=self.ref_path,
                verify_after_action=self.verify_after_action,
                logger_callback=engine_logger,
                restore_workspace=True,
            )

            if not self._is_cancelled:
                self.result_signal.emit(results)

        except Exception as e:
            self.log_signal.emit(f"Worker Exception: {e!s}", "ERROR")
        finally:
            self.finished_signal.emit()


class MagnifierLabel(QLabel):
    """Show a magnified region under the cursor with crosshair overlays."""

    def __init__(
        self,
        parent: QWidget | None = None,
        zoom: float = 3.0,
        size: int = 300,
    ) -> None:
        """Initialize the magnifier with mouse tracking enabled."""
        super().__init__(parent)
        self.setMouseTracking(True)
        self.zoom = zoom
        self.mag_size = size
        self.full_res_frame: QPixmap | None = None
        self.crosshairs: list = []

    def set_frame(self, pixmap: QPixmap) -> None:
        """Store the original resolution pixmap and trigger a redraw."""
        self.full_res_frame = pixmap
        self.update()

    def update_crosshairs(self, nodes: list) -> None:
        """Update the list of coordinates to draw on the overlay."""
        self.crosshairs = nodes
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Repaint the widget on mouse movement to update the magnifier position."""
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        """Clear the magnifier loupe from the display when the cursor exits."""
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event: QPaintEvent) -> None:
        """Render the pre-annotated debug image and the interactive magnifier."""
        if self.full_res_frame is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Coordinate Setup
        pw, ph = self.full_res_frame.width(), self.full_res_frame.height()
        sw, sh = self.width(), self.height()
        scale = min(sw / pw, sh / ph)
        dx, dy = (sw - int(pw * scale)) // 2, (sh - int(ph * scale)) // 2

        # 2. Draw Background
        painter.drawPixmap(
            dx,
            dy,
            int(pw * scale),
            int(ph * scale),
            self.full_res_frame,
        )

        # 3. Magnifier Logic
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        if self.underMouse():
            img_x = int((cursor_pos.x() - dx) / scale)
            img_y = int((cursor_pos.y() - dy) / scale)

            if 0 <= img_x < pw and 0 <= img_y < ph:
                cw = int(self.mag_size / self.zoom)
                crop_x = max(0, min(img_x - cw // 2, pw - cw))
                crop_y = max(0, min(img_y - cw // 2, ph - cw))
                cropped = self.full_res_frame.copy(crop_x, crop_y, cw, cw)
                zoomed = cropped.scaled(
                    self.mag_size,
                    self.mag_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

                # Determine Loupe Position (Avoid edges)
                tx, ty = cursor_pos.x() + 25, cursor_pos.y() + 25
                if tx + self.mag_size > sw:
                    tx = cursor_pos.x() - self.mag_size - 25
                if ty + self.mag_size > sh:
                    ty = cursor_pos.y() - self.mag_size - 25

                # Draw the zoomed-in portion (with burned-in annotations)
                painter.drawPixmap(tx, ty, zoomed)

                # 4. Loupe Cosmetics
                # Border
                painter.setPen(QPen(QColor(ACCENT), 2))
                painter.drawRect(tx, ty, self.mag_size, self.mag_size)

                # Coordinates Text with Shadow for visibility
                text = f"IMG: {img_x}, {img_y}"
                painter.setPen(QColor(0, 0, 0))  # Shadow
                painter.drawText(tx + 6, ty + 16, text)
                painter.setPen(QColor("white"))  # Actual Text
                painter.drawText(tx + 5, ty + 15, text)

        painter.end()


class GroundingLab(QMainWindow):
    """Operate the interactive dashboard for AI grounding coordinates."""

    def __init__(self) -> None:
        """Initialize the lab environment and AI engine."""
        super().__init__()
        self.engine = AiGroundingEngine()
        self.worker: AIWorker | None = None
        self.ref_image_path: str | None = None
        self.node_data: list = []

        self.setWindowTitle("AI GROUNDING LAB")
        self.resize(1400, 900)
        self._setup_ui()

    # --------------------------
    # UI Setup
    # --------------------------
    def _setup_ui(self) -> None:
        """Create layout, styles, and widget tree."""
        self._apply_styles()
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        layout.addWidget(self._create_sidebar())
        layout.addWidget(self._create_viewport_area(), stretch=1)

    def _apply_styles(self) -> None:
        """Apply the cyberpunk-themed QSS stylesheet."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BG_DARK};
            }}
            QWidget {{
                color: #E0E6ED;
                font-family: 'Segoe UI Semibold';
                font-size: 13px;
            }}
            QGroupBox {{
                border: 1px solid #2D333F;
                border-radius: 8px;
                margin-top: 25px;
                font-weight: bold;
                color: {ACCENT};
                text-transform: uppercase;
            }}
            QGroupBox:title {{
                top: -10px;
            }}
            QLineEdit, QComboBox {{
                background-color: {BG_DARK};
                border: 1px solid #2D333F;
                padding: 10px;
                border-radius: 5px;
                color: white;
            }}
            QPushButton {{
                background-color: {SURFACE};
                border: 1px solid #2D333F;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {ACCENT};
                background-color: #1C222D;
            }}
            QTextEdit {{
                background-color: #050505;
                border: none;
                font-family: 'Consolas';
                font-size: 12px;
                border-left: 2px solid {ACCENT};
            }}
        """)

    def _create_sidebar(self) -> QFrame:
        """Construct the control sidebar and logs panel."""
        sidebar = QFrame()
        sidebar.setFixedWidth(400)
        layout = QVBoxLayout(sidebar)

        # Target Scope
        scope_group = QGroupBox("1. Target Scope")
        s_layout = QVBoxLayout(scope_group)
        scope_header = QHBoxLayout()
        self.window_selector = QComboBox()
        self.refresh_windows()
        btn_refresh = QPushButton("⟳")
        btn_refresh.setFixedWidth(40)
        btn_refresh.clicked.connect(self.refresh_windows)
        scope_header.addWidget(self.window_selector)
        scope_header.addWidget(btn_refresh)
        s_layout.addLayout(scope_header)
        layout.addWidget(scope_group)

        # Instruction
        task_group = QGroupBox("2. Instruction")
        t_layout = QVBoxLayout(task_group)
        self.instruction_input = QLineEdit()
        self.instruction_input.setPlaceholderText("Search for...")
        self.checkbox_verify = QCheckBox("Verify element location after action")
        t_layout.addWidget(self.instruction_input)
        t_layout.addWidget(self.checkbox_verify)
        layout.addWidget(task_group)

        # Visual Anchor
        anchor_group = QGroupBox("3. Visual Anchor")
        a_layout = QVBoxLayout(anchor_group)
        btn_row_layout = QHBoxLayout()
        btn_ref = QPushButton("LOAD")
        btn_ref.clicked.connect(self.select_reference)
        btn_clear_ref = QPushButton("CLEAR")
        btn_clear_ref.clicked.connect(self.clear_anchor)
        btn_row_layout.addWidget(btn_ref)
        btn_row_layout.addWidget(btn_clear_ref)
        self.ref_preview = QLabel("NO IMAGE")
        self.ref_preview.setFixedSize(60, 60)
        self.ref_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ref_preview.setStyleSheet("border: 1px dashed #333; background: #000;")
        a_layout.addLayout(btn_row_layout)
        a_layout.addWidget(self.ref_preview)
        layout.addWidget(anchor_group)

        # Logs
        self.reasoning_view = QTextEdit()
        self.reasoning_view.setReadOnly(True)
        layout.addWidget(QLabel("REASONING PIPELINE"))
        layout.addWidget(self.reasoning_view)

        # Controls
        ctrl_layout = QHBoxLayout()
        self.btn_run = QPushButton("RUN")
        self.btn_run.setStyleSheet(f"background-color: {ACCENT}; color: black;")
        self.btn_run.clicked.connect(self.start_diagnostic)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.cancel_diagnostic)

        ctrl_layout.addWidget(self.btn_run)
        ctrl_layout.addWidget(self.btn_stop)
        layout.addLayout(ctrl_layout)

        return sidebar

    def _create_viewport_area(self) -> QWidget:
        """Create the primary visual display for screen captures and table below."""
        container = QWidget()
        layout = QVBoxLayout(container)

        # Image display with magnifier
        self.viewport = MagnifierLabel(zoom=3, size=400)
        self.viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewport.setStyleSheet("background-color: #000; border: 1px solid #333;")
        layout.addWidget(self.viewport, stretch=1)

        # Move results table below viewport
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ["COORDS [X,Y]", "SCORE", "AREA", "NEIGHBORS", "RANK"],
        )
        h_header = self.results_table.horizontalHeader()
        if h_header:
            h_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            h_header.setDefaultAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
        self.results_table.setFixedHeight(180)
        layout.addWidget(self.results_table)

        return container

    # --------------------------
    # Window Management
    # --------------------------
    def refresh_windows(self) -> None:
        """Update dropdown with current OS window titles."""
        current_selection = self.window_selector.currentText()
        titles = [w.title for w in gw.getAllWindows() if w.title.strip()]
        self.window_selector.clear()
        items = ["Desktop", *sorted(set(titles))]
        self.window_selector.addItems(items)
        if current_selection in items:
            self.window_selector.setCurrentText(current_selection)

    # --------------------------
    # Logging
    # --------------------------
    def log_message(self, msg: str, level: str = "INFO") -> None:
        """Append formatted logs to the console."""
        color = {"INFO": ACCENT, "SUCCESS": SUCCESS, "ERROR": ERROR}.get(level, "white")
        self.reasoning_view.append(
            f"<span style='color:{color};'>[{level}]</span> {msg}",
        )

    # --------------------------
    # Worker Control
    # --------------------------
    def start_diagnostic(self) -> None:
        """Initiate the AI grounding worker thread."""
        instr = self.instruction_input.text()
        if not instr:
            return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_stop.setText("STOP")
        self.reasoning_view.clear()
        self.results_table.setRowCount(0)
        self.node_data = []

        self.worker = AIWorker(
            self.engine,
            instr,
            self.window_selector.currentText(),
            self.ref_image_path,
            verify_after_action=self.checkbox_verify.isChecked(),
        )
        self.worker.log_signal.connect(self.log_message)
        self.worker.result_signal.connect(self.process_results)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def cancel_diagnostic(self) -> None:
        """Force stop the worker thread."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.terminate()
            self.worker.wait()
            self.log_message("Operation forcefully terminated.", "ERROR")
            self.on_finished()

    def on_finished(self) -> None:
        """Reset UI state after thread completion."""
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("STOP")

    # --------------------------
    # Result Handling
    # --------------------------
    def process_results(self, results: list[UIElementNode]) -> None:
        """Update display using the pre-rendered DEBUG image from the engine."""
        display_image = getattr(self.engine, "last_debug_image", None)

        if display_image is None:
            self.log_message("Visualization failed: No debug image found.", "ERROR")
            return

        # Convert the annotated PIL image to QPixmap
        q_img = ImageQt.ImageQt(display_image.convert("RGBA"))
        pixmap = QPixmap.fromImage(q_img)

        self.viewport.set_frame(pixmap)
        self._update_results_table(results)

    def _update_results_table(self, results: list[UIElementNode]) -> None:
        """Refresh the UI table with raw data."""
        self.results_table.setRowCount(0)
        for i, node in enumerate(results):
            px, py = node["coords"]
            self.results_table.insertRow(i)
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{px}, {py}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{node['score']:.2f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(node["area"]))
            self.results_table.setItem(
                i,
                3,
                QTableWidgetItem(", ".join(node["neighbors"])),
            )
            self.results_table.setItem(i, 4, QTableWidgetItem(str(node["rank"])))

    # --------------------------
    # Reference Image
    # --------------------------
    def select_reference(self) -> None:
        """Load a reference anchor via file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Anchor",
            "",
            "Images (*.png *.jpg)",
        )
        if path:
            self.ref_image_path = path
            scaled_pix = QPixmap(path).scaled(
                60,
                60,
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            self.ref_preview.setPixmap(scaled_pix)

    def clear_anchor(self) -> None:
        """Remove current visual anchor and clear preview."""
        self.ref_image_path = None
        self.ref_preview.setText("NO IMAGE")
        self.ref_preview.setPixmap(QPixmap())

    # --------------------------
    # Close Event
    # --------------------------
    def closeEvent(self, event: QEvent) -> None:
        """Clean up resources when closing the lab."""
        event.accept()


def run() -> None:
    """Execute the main application loop."""
    set_high_dpi_awareness()
    app = QApplication(sys.argv)
    window = GroundingLab()
    window.showMaximized()
    sys.exit(app.exec())
