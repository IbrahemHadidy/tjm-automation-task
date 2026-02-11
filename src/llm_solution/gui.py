"""Provide a diagnostic laboratory interface for AI grounding and vision tasks."""

import os
import sys
from pathlib import Path

import pygetwindow as gw
from PyQt6.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
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

from llm_solution.grounding_engine import AiGroundingEngine, UIElementNode

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

    log_signal = pyqtSignal(str, str)
    result_signal = pyqtSignal(list)
    finished_signal = pyqtSignal()

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
        """Perform custom rendering of the image and the magnifier loupe."""
        if self.full_res_frame is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Scale image to fit the label while maintaining aspect ratio
        pw, ph = self.full_res_frame.width(), self.full_res_frame.height()
        sw, sh = self.width(), self.height()
        scale = min(sw / pw, sh / ph)

        view_w, view_h = int(pw * scale), int(ph * scale)
        dx = (sw - view_w) // 2
        dy = (sh - view_h) // 2

        # 2. Draw the background image
        painter.drawPixmap(dx, dy, view_w, view_h, self.full_res_frame)

        # 3. Draw Crosshairs
        pen = QPen(QColor(SUCCESS), 2)
        painter.setPen(pen)
        for node in self.crosshairs:
            coords = node.get("coords", [0, 0])
            rx, ry = coords[0], coords[1]
            px = int(rx * scale) + dx
            py = int(ry * scale) + dy
            painter.drawLine(px - 10, py, px + 10, py)
            painter.drawLine(px, py - 10, px, py + 10)

        # 4. Magnifier Logic
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        if self.underMouse():
            # Convert widget coordinates to source image coordinates
            img_x = int((cursor_pos.x() - dx) / scale)
            img_y = int((cursor_pos.y() - dy) / scale)

            if 0 <= img_x < pw and 0 <= img_y < ph:
                cw = int(self.mag_size / self.zoom)
                # Crop area from the original high-res image
                crop_x = max(0, min(img_x - cw // 2, pw - cw))
                crop_y = max(0, min(img_y - cw // 2, ph - cw))

                cropped = self.full_res_frame.copy(crop_x, crop_y, cw, cw)
                zoomed = cropped.scaled(
                    self.mag_size,
                    self.mag_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

                # Offset the loupe so it's not directly under the cursor
                tx = cursor_pos.x() + 25
                ty = cursor_pos.y() + 25

                # Flip position if it goes off screen
                if tx + self.mag_size > sw:
                    tx = cursor_pos.x() - self.mag_size - 25
                if ty + self.mag_size > sh:
                    ty = cursor_pos.y() - self.mag_size - 25

                painter.drawPixmap(tx, ty, zoomed)
                painter.setPen(QPen(QColor(ACCENT), 2))
                painter.drawRect(tx, ty, self.mag_size, self.mag_size)
                painter.setPen(QColor("white"))
                painter.drawText(tx + 5, ty + 15, f"IMAGE: {img_x}, {img_y}")

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

        self.setWindowTitle("AI GROUNDING LAB v0.1")
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
        items = ["Entire Desktop", *sorted(set(titles))]
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
        """Visualize AI coordinates using bounding boxes instead of crosshairs."""
        self.log_message(f"Mapped {len(results)} targets.", "SUCCESS")
        self.results_table.setRowCount(0)
        self.node_data = results.copy()

        vision_path = Path("ai_vision_input.png")
        if not vision_path.exists():
            self.log_message("Visual capture file missing.", "ERROR")
            return

        pixmap = QPixmap(str(vision_path))
        painter = QPainter(pixmap)
        pen = QPen(QColor(SUCCESS))
        pen.setWidth(3)
        painter.setPen(pen)
        font = QFont("Consolas", 12)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)

        for i, node in enumerate(results):
            px, py = node["coords"]
            score = node["score"]
            area = node["area"]
            neighbors = ", ".join(node["neighbors"])
            rank = node["rank"]
            width, height = node["size"] or (20, 20)

            # Draw bounding box centered on coords
            painter.drawRect(px - width // 2, py - height // 2, width, height)

            # Draw rank/label near box
            painter.drawText(
                px + width // 2 + 5,
                py + height // 2 + 5,
                f"Rank:{rank}",
            )

            # Update results table
            self.results_table.insertRow(i)
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{px}, {py}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{score:.2f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(area))
            self.results_table.setItem(i, 3, QTableWidgetItem(neighbors))
            self.results_table.setItem(i, 4, QTableWidgetItem(str(rank)))

            # Log for reasoning/debugging
            self.log_message(
                f"Node {i + 1}: {area} | Neighbors: {neighbors} | Score: {score:.2f} | Rank: {rank}",
            )

        painter.end()

        # Update magnifier overlay
        self.viewport.set_frame(pixmap)
        self.viewport.update_crosshairs(self.node_data)

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
        """Clean up temporary AI artifacts when closing the lab."""
        self.engine.cleanup()
        event.accept()


def run() -> None:
    """Execute the main application loop."""
    app = QApplication(sys.argv)
    window = GroundingLab()
    window.showMaximized()
    sys.exit(app.exec())
