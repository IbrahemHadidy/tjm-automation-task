"""Provide a diagnostic laboratory interface for AI grounding and vision tasks."""

import ctypes
import os
import sys
from pathlib import Path

import pygetwindow as gw
from PyQt6.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
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

from llm_solution.grounding_engine import AiGroundingEngine

# High-DPI Scaling Configuration
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"
os.environ["QT_API"] = "pyqt6"

# Cyberpunk UI Constants
BG_DARK = "#0B0E14"
SURFACE = "#151921"
ACCENT = "#00F0FF"
SUCCESS = "#39FF14"
ERROR = "#FF3131"


class AIWorker(QThread):
    """Handle AI coordinate resolution in a background thread."""

    log_signal = pyqtSignal(str, str)
    result_signal = pyqtSignal(list)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        engine: AiGroundingEngine,
        instruction: str,
        target_window: str,
        ref_path: str | None,
    ) -> None:
        """Initialize the worker thread with task parameters."""
        super().__init__()
        self.engine = engine
        self.instruction = instruction
        self.target_window = target_window
        self.ref_path = ref_path
        self._is_cancelled = False

    def cancel(self) -> None:
        """Flag the worker to stop processing."""
        self._is_cancelled = True

    def run(self) -> None:
        """Execute the AI resolution pipeline."""
        try:
            if self._is_cancelled:
                return

            def engine_logger(msg: str) -> None:
                """Clean tags and emit engine logs to the UI via signals."""
                if not self._is_cancelled:
                    # Determine appropriate log level based on engine tags
                    level = "INFO"
                    if "[SUCCESS]" in msg:
                        level = "SUCCESS"
                    elif "[ERROR]" in msg:
                        level = "ERROR"

                    # Strip tags to prevent nested logs like [INFO] [SUCCESS]
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
                logger_callback=engine_logger,
            )

            if not self._is_cancelled:
                self.result_signal.emit(results)
        except Exception as e:
            self.log_signal.emit(f"Worker Exception: {e!s}", "ERROR")
        finally:
            self.finished_signal.emit()


class GroundingLab(QMainWindow):
    """Operate the interactive dashboard for visualizing AI grounding coordinates."""

    def __init__(self) -> None:
        """Set up the lab environment and initialize the AI engine."""
        super().__init__()
        self.engine = AiGroundingEngine()
        self.worker: AIWorker | None = None
        self.ref_image_path: str | None = None

        self.setWindowTitle("AI GROUNDING LAB v0.1")
        self.resize(1400, 900)
        self.setup_ui()

    def setup_ui(self) -> None:
        """Initialize the layout, styles, and widget tree."""
        self._apply_styles()
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        sidebar = self._create_sidebar()
        viewport = self._create_viewport_area()

        layout.addWidget(sidebar)
        layout.addWidget(viewport, stretch=1)

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

    def _create_sidebar(self) -> QFrame:  # noqa: PLR0915
        """Construct the control sidebar and logs panel."""
        sidebar = QFrame()
        sidebar.setFixedWidth(400)
        layout = QVBoxLayout(sidebar)

        # 1. Target Scope
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

        # 2. Instruction
        task_group = QGroupBox("2. Instruction")
        t_layout = QVBoxLayout(task_group)
        self.instruction_input = QLineEdit()
        self.instruction_input.setPlaceholderText("Search for...")
        t_layout.addWidget(self.instruction_input)
        layout.addWidget(task_group)

        # 3. Visual Anchor
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

        # 4. Results Table
        layout.addWidget(QLabel("MAPPED COORDINATES"))
        self.results_table = QTableWidget(0, 2)
        self.results_table.setHorizontalHeaderLabels(["COORDS [X, Y]", "SCORE"])
        h_header = self.results_table.horizontalHeader()
        if h_header:
            h_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setFixedHeight(180)
        layout.addWidget(self.results_table)

        # 5. Logs
        self.reasoning_view = QTextEdit()
        self.reasoning_view.setReadOnly(True)
        layout.addWidget(QLabel("REASONING PIPELINE"))
        layout.addWidget(self.reasoning_view)

        # 6. Controls
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
        """Create the primary visual display for screen captures."""
        container = QWidget()
        layout = QVBoxLayout(container)
        self.viewport = QLabel("IDLE")
        self.viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewport.setStyleSheet("background-color: #000; border: 1px solid #333;")
        layout.addWidget(self.viewport)
        return container

    def refresh_windows(self) -> None:
        """Update the dropdown with current OS window titles."""
        current_selection = self.window_selector.currentText()
        titles = [w.title for w in gw.getAllWindows() if w.title.strip()]
        self.window_selector.clear()
        items = ["Entire Desktop", *sorted(set(titles))]
        self.window_selector.addItems(items)
        if current_selection in items:
            self.window_selector.setCurrentText(current_selection)

    def log_message(self, msg: str, level: str = "INFO") -> None:
        """Append formatted logs to the console."""
        color = {"INFO": ACCENT, "SUCCESS": SUCCESS, "ERROR": ERROR}.get(level, "white")
        self.reasoning_view.append(
            f"<span style='color:{color};'>[{level}]</span> {msg}",
        )

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

        self.worker = AIWorker(
            self.engine,
            instr,
            self.window_selector.currentText(),
            self.ref_image_path,
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

    def process_results(self, results: list) -> None:
        """Visualize AI coordinates directly using pixel values from the engine."""
        self.log_message(f"Mapped {len(results)} targets.", "SUCCESS")
        self.results_table.setRowCount(0)

        # Load the screenshot created by the engine
        vision_path = Path("ai_vision_input.png")
        if not vision_path.exists():
            self.log_message("Visual capture file missing.", "ERROR")
            return

        pixmap = QPixmap(str(vision_path))
        painter = QPainter(pixmap)
        pen = QPen(QColor(SUCCESS))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setFont(QFont("Consolas", 12, QFont.Weight.Bold))

        for i, node in enumerate(results):
            # Coordinates are now ALREADY in pixels thanks to the engine update
            px, py = node.get("coords", [0, 0])
            score = node.get("score", 0.0)

            # Draw Crosshair using the pixel coordinates directly
            painter.drawLine(px - 15, py, px + 15, py)
            painter.drawLine(px, py - 15, px, py + 15)
            painter.drawText(px + 8, py - 8, f"ID:{i + 1} [{score}]")

            # Update Table
            self.results_table.insertRow(i)
            self.results_table.setItem(
                i,
                0,
                QTableWidgetItem(f"{px}, {py}"),
            )
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{score:.2f}"))

        painter.end()

        # Scale and display in viewport
        scaled_pix = pixmap.scaled(
            self.viewport.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.viewport.setPixmap(scaled_pix)

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
        """Remove the current visual anchor and clear preview."""
        self.ref_image_path = None
        self.ref_preview.setText("NO IMAGE")
        self.ref_preview.setPixmap(QPixmap())

    def closeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Ensure all temporary AI artifacts are deleted when closing the lab."""
        self.engine.cleanup()
        event.accept()


def run() -> None:
    """Execute the main application loop."""
    app = QApplication(sys.argv)
    window = GroundingLab()
    window.showMaximized()
    sys.exit(app.exec())

