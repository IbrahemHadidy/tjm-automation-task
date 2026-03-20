"""FSM Telemetry Viewer.

This module provides a desktop viewer for telemetry `metadata.json` manifests
produced by the FSM automation system. It displays:

- Execution timeline
- Step thumbnails
- Zoomable image preview with a magnifier
- Run-level metrics and summary
- Export video functionality (requires optional monitoring module)
"""

from __future__ import annotations

import contextlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QEvent,
    QPoint,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QIcon,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# Optional monitoring helper
_monitoring_module: Any = None
try:
    import monitoring  # type: ignore[import-not-found]

    _monitoring_module = monitoring
except Exception:
    _monitoring_module = None

# --- Style constants ---
BG_COLOR = "#0F1113"
SURFACE_COLOR = "#151719"
ACCENT_COLOR = "#00A8FF"
SUCCESS_COLOR = "#2EC4B6"
WARNING_COLOR = "#FFB400"
ERROR_COLOR = "#FF5C5C"
TEXT_COLOR = "#E6EEF3"
MUTED_COLOR = "#9AA6B2"

GLOBAL_STYLE = f"""
    QMainWindow {{ background-color: {BG_COLOR}; }}
    QWidget {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Segoe UI', 'Consolas';
        font-size: 13px;
    }}
    QToolBar {{
        background: {SURFACE_COLOR};
        spacing: 8px;
        padding: 6px;
        border-bottom: 1px solid #1F2426;
    }}
    QPushButton, QComboBox, QLineEdit {{
        background-color: transparent;
        border: 1px solid #2A2F31;
        border-radius: 6px;
        padding: 6px 8px;
    }}
    QLineEdit {{ min-width: 200px; }}
    QLabel#run_label {{ font-weight: bold; color: {ACCENT_COLOR}; }}
    QListWidget {{
        background-color: #0A0B0C;
        border: 1px solid #1D2224;
        border-radius: 6px;
        outline: none;
    }}
    QListWidget::item {{ padding: 8px; border-bottom: 1px solid #0F1314; }}
    QListWidget::item:selected {{
        background-color: #081018;
        border-left: 4px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
    }}
    QTextEdit {{
        background-color: #070808;
        border: 1px solid #1C2021;
        font-family: 'Consolas';
        font-size: 12px;
        padding: 8px;
        border-radius: 6px;
    }}
    QGroupBox {{
        border: 1px solid #212526;
        border-radius: 8px;
        margin-top: 14px;
        padding: 8px;
        color: {MUTED_COLOR};
        font-weight: bold;
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; top: 2px; }}
    QStatusBar {{ background: transparent; color: {MUTED_COLOR}; }}
"""


class SpinnerOverlay(QWidget):
    """A full-window semi-transparent overlay with a simple rotating spinner.

    Use .start() to show & run the spinner, and .stop() to hide it.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize overlay spinner widget."""
        super().__init__(parent)
        # Cover the parent completely
        if parent:
            self.setGeometry(parent.rect())
        # Overlay should be on top
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)  # noqa: FBT003
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)  # noqa: FBT003
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._angle = 0.0  # degrees
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 FPS
        self._timer.timeout.connect(self._advance)

        # Visual tuning
        self._bg_color = QColor(0, 0, 0, 160)  # semi-transparent black
        self._spinner_color = QColor(ACCENT_COLOR)
        self._spinner_width = 6
        self._arc_span_deg = 120.0

        self.hide()

    def start(self) -> None:
        """Show overlay and start animation."""
        self._angle = 0.0
        self.setVisible(True)
        self.raise_()
        self._timer.start()

    def stop(self) -> None:
        """Stop animation and hide overlay."""
        self._timer.stop()
        self.setVisible(False)

    def _advance(self) -> None:
        self._angle = (self._angle + 10.0) % 360.0
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: ARG002
        """Paint translucent background and centered rotating arc."""
        if not self.isVisible():
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # background
        p.fillRect(self.rect(), self._bg_color)

        # spinner geometry
        side = min(self.width(), self.height()) // 6
        side = max(side, 24)
        center_x = self.width() // 2
        center_y = self.height() // 2
        r = side
        rect = (
            center_x - r,
            center_y - r,
            2 * r,
            2 * r,
        )

        pen = QPen(self._spinner_color)
        pen.setWidth(self._spinner_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)

        # QPainter.drawArc uses 1/16th degrees
        start_angle = int(self._angle * 16)
        span_angle = int(self._arc_span_deg * 16)
        p.drawArc(*rect, start_angle, span_angle)

        p.end()


class ZoomableImageLabel(QLabel):
    """A QLabel-based image viewer with a movable magnifier."""

    def __init__(self, text: str = "NO IMAGE LOADED") -> None:
        """Create a zoomable image label."""
        super().__init__(text)
        self.setMouseTracking(True)
        self.full_res_pixmap: QPixmap | None = None
        self.mouse_pos: QPoint | None = None
        self.zoom_factor = 2.8
        self.zoom_size = 240
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def load_image(self, path: str) -> None:
        """Load a full-resolution image from `path`."""
        if not path:
            self.full_res_pixmap = None
            self.setText("NO IMAGE PATH")
            return

        p = Path(path)
        if not p.exists():
            self.full_res_pixmap = None
            self.setText("IMAGE FILE NOT FOUND")
            return

        pix = QPixmap(str(p))
        if pix.isNull():
            self.full_res_pixmap = None
            self.setText("FAILED TO LOAD IMAGE")
            return

        self.full_res_pixmap = pix
        self.update_display()

    def update_display(self) -> None:
        """Scale the loaded pixmap to fit the widget."""
        if self.full_res_pixmap:
            scaled = self.full_res_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
        else:
            self.setPixmap(QPixmap())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Track mouse position for magnifier rendering."""
        self.mouse_pos = event.pos()
        self.update()
        super().mouseMoveEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the base image and the magnifier overlay."""
        super().paintEvent(event)
        if not self.full_res_pixmap or not self.mouse_pos or not self.pixmap():
            return

        displayed_pixmap = self.pixmap()
        dx = (self.width() - displayed_pixmap.width()) // 2
        dy = (self.height() - displayed_pixmap.height()) // 2

        mx = self.mouse_pos.x() - dx
        my = self.mouse_pos.y() - dy

        if 0 <= mx <= displayed_pixmap.width() and 0 <= my <= displayed_pixmap.height():
            scale_x = self.full_res_pixmap.width() / displayed_pixmap.width()
            scale_y = self.full_res_pixmap.height() / displayed_pixmap.height()

            full_x = int(mx * scale_x)
            full_y = int(my * scale_y)

            crop_w = int(self.zoom_size / self.zoom_factor)
            x0 = max(0, full_x - crop_w // 2)
            y0 = max(0, full_y - crop_w // 2)
            x0 = min(x0, self.full_res_pixmap.width() - crop_w)
            y0 = min(y0, self.full_res_pixmap.height() - crop_w)

            try:
                crop_rect = self.full_res_pixmap.copy(x0, y0, crop_w, crop_w)
            except Exception:
                return

            zoom_p = crop_rect.scaled(
                self.zoom_size,
                self.zoom_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )

            p = QPainter(self)
            tx = self.mouse_pos.x() + 18
            ty = self.mouse_pos.y() + 18
            if tx + self.zoom_size > self.width():
                tx = self.mouse_pos.x() - self.zoom_size - 18
            if ty + self.zoom_size > self.height():
                ty = self.mouse_pos.y() - self.zoom_size - 18

            p.drawPixmap(tx, ty, zoom_p)
            p.setPen(QPen(QColor(ACCENT_COLOR), 2))
            p.drawRect(tx, ty, self.zoom_size, self.zoom_size)
            p.end()

    def leaveEvent(self, event: QEvent) -> None:
        """Clear magnifier when the mouse leaves."""
        self.mouse_pos = None
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Rescale visible pixmap on resize."""
        self.update_display()
        super().resizeEvent(event)


class ManifestLoaderThread(QThread):
    """Load metadata.json in the background to keep UI responsive."""

    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, path: Path) -> None:
        """Initialize the manifest loader thread."""
        super().__init__()
        self.path = path

    def run(self) -> None:
        """Load JSON from `self.path` and emit signals on success or failure.

        Emits:
        finished (dict): Emitted with loaded JSON data on success.
        failed (str): Emitted with error message if loading fails.
        """
        try:
            with self.path.open(encoding="utf-8") as f:
                data = json.load(f)
            self.finished.emit(data)
        except Exception as e:
            self.failed.emit(str(e))


class ThumbnailLoaderThread(QThread):
    """Background thread to load and scale thumbnail images as QImage.

    Emits a list of dicts: {"index": idx, "qimage": QImage} so the main
    thread can convert to QPixmap safely and set icons.
    """

    finished = Signal(list)

    def __init__(
        self,
        steps: list[dict[str, Any]],
        icon_size: QSize = QSize(160, 90),  # noqa: B008
    ) -> None:
        """Initialize the thumbnail generation worker."""
        super().__init__()
        self.steps = list(steps)
        self.icon_size = icon_size

    def run(self) -> None:
        """Process all steps to generate scaled thumbnails."""
        out: list[dict[str, Any]] = []
        for step in self.steps:
            try:
                idx = int(step.get("index", 0))
            except Exception:
                idx = 0
            ppath = step.get("screenshot_path", "")
            if ppath and Path(ppath).exists():
                img = QImage(ppath)
                if not img.isNull():
                    scaled = img.scaled(
                        self.icon_size.width(),
                        self.icon_size.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    out.append({"index": idx, "qimage": scaled})
                    continue
            # placeholder: append None to allow mapping back
            out.append({"index": idx, "qimage": None})
        self.finished.emit(out)


class TelemetryViewer(QMainWindow):
    """Main telemetry dashboard window."""

    def __init__(self) -> None:
        """Build the telemetry viewer UI."""
        super().__init__()
        self.setWindowTitle("FSM Telemetry Viewer")
        self.setMinimumSize(1200, 820)
        self.setStyleSheet(GLOBAL_STYLE)

        self.current_run_data: dict[str, Any] = {}
        self.filtered_steps: list[dict[str, Any]] = []
        self._monitoring_module = _monitoring_module

        # cache mapping step index -> QListWidgetItem for thumbnails
        self._thumb_items_by_index: dict[int, QListWidgetItem] = {}

        self.init_ui()

    def init_ui(self) -> None:
        """Create and wire UI widgets."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ---------- Toolbar ----------
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.btn_load = QPushButton("Load metadata.json")
        self.btn_load.clicked.connect(self.load_manifest)
        toolbar.addWidget(self.btn_load)

        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(lambda: self.navigate_step(-1))
        toolbar.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(lambda: self.navigate_step(1))
        toolbar.addWidget(self.btn_next)

        toolbar.addSeparator()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by state or action...")
        self.search_input.textChanged.connect(self.on_search_changed)
        toolbar.addWidget(self.search_input)

        self.state_filter = QComboBox()
        self.state_filter.addItem("All states")
        self.state_filter.currentTextChanged.connect(self.on_filter_changed)
        toolbar.addWidget(self.state_filter)

        toolbar.addSeparator()

        self.btn_export = QPushButton("Open Video")
        self.btn_export.clicked.connect(self.open_video)
        self.btn_export.setEnabled(self._monitoring_module is not None)
        toolbar.addWidget(self.btn_export)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.run_label = QLabel("NO RUN LOADED")
        self.run_label.setObjectName("run_label")
        toolbar.addWidget(self.run_label)

        # ---------- Splitter with three columns ----------
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT: Timeline (top) + Selected Step (bottom) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)
        left_panel.setMinimumWidth(320)

        timeline_group = QGroupBox("Execution Timeline")
        timeline_layout = QVBoxLayout(timeline_group)
        self.timeline_list = QListWidget()
        self.timeline_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection,
        )
        self.timeline_list.currentItemChanged.connect(self.on_step_selected)
        self.timeline_list.setUniformItemSizes(True)
        timeline_layout.addWidget(self.timeline_list)
        left_layout.addWidget(timeline_group, stretch=3)

        nav_group = QGroupBox("Selected Step")
        nav_layout = QVBoxLayout(nav_group)
        self.lbl_selected = QLabel("No step selected")
        nav_layout.addWidget(self.lbl_selected)
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setEnabled(False)
        self.scrub.valueChanged.connect(self.timeline_list.setCurrentRow)
        nav_layout.addWidget(self.scrub)
        left_layout.addWidget(nav_group, stretch=0)

        splitter.addWidget(left_panel)

        # --- CENTER: Image viewer + Step Context ---
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(6, 6, 6, 6)
        center_layout.setSpacing(8)

        self.image_display = ZoomableImageLabel()
        self.image_display.setStyleSheet("background: #050505; border: 1px solid #222;")
        center_layout.addWidget(self.image_display, stretch=5)

        details_group = QGroupBox("Step Context")
        details_layout = QVBoxLayout(details_group)
        self.txt_details = QTextEdit()
        self.txt_details.setReadOnly(True)
        self.txt_details.setMaximumHeight(220)
        self.txt_details.setMinimumHeight(220)
        details_layout.addWidget(self.txt_details)
        center_layout.addWidget(details_group, stretch=2)

        splitter.addWidget(center_panel)

        # --- RIGHT: Thumbnails (top) + Run Summary (bottom) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(8)
        right_panel.setMinimumWidth(320)

        thumbs_group = QGroupBox("Thumbnails")
        thumbs_layout = QVBoxLayout(thumbs_group)
        self.thumb_list = QListWidget()
        self.thumb_list.setViewMode(QListView.ViewMode.ListMode)
        self.thumb_list.setIconSize(QSize(160, 100))
        self.thumb_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.thumb_list.setMovement(QListView.Movement.Static)
        self.thumb_list.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.thumb_list.setWrapping(False)
        self.thumb_list.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.thumb_list.currentItemChanged.connect(self.on_thumb_selected)
        thumbs_layout.addWidget(self.thumb_list)
        right_layout.addWidget(thumbs_group, stretch=2)  # 2/3 of right column

        metrics_group = QGroupBox("Run Summary")
        metrics_layout = QVBoxLayout(metrics_group)
        self.lbl_metrics = QLabel("")
        self.lbl_metrics.setWordWrap(True)
        metrics_layout.addWidget(self.lbl_metrics)
        self.metrics_details = QTextEdit()
        self.metrics_details.setReadOnly(True)
        self.metrics_details.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        metrics_layout.addWidget(self.metrics_details)
        right_layout.addWidget(metrics_group, stretch=1)  # 1/3 of right column

        splitter.addWidget(right_panel)

        # Splitter sizing / behaviour (left, center, right)
        splitter.setSizes([340, 720, 360])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setChildrenCollapsible(False)

        main_layout.addWidget(splitter)

        # ---------- Status ----------
        status = QStatusBar()
        status.showMessage("Ready — load a metadata.json to begin.")
        self.setStatusBar(status)
        self.status = status

        # overlay spinner that covers the whole central area
        central_widget = self.centralWidget()
        self.overlay = SpinnerOverlay(central_widget)
        self.overlay.hide()

        # placeholder for currently running thumb thread
        self._thumb_thread: ThumbnailLoaderThread | None = None

    def format_ts(self, unix_ts: float | None) -> str:
        """Return a human-readable timestamp for a UNIX epoch value."""
        if not unix_ts:
            return "N/A"
        try:
            ts = float(unix_ts)
            if ts > 1e12:
                ts = ts / 1000.0
            dt = datetime.fromtimestamp(ts, tz=UTC).astimezone()
            return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        except Exception:
            return str(unix_ts)

    def load_manifest(self) -> None:
        """Open a file dialog and load `metadata.json` in a background thread."""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select metadata.json",
            "",
            "JSON Files (*.json)",
        )
        if not path_str:
            return

        path = Path(path_str)

        # Show overlay spinner
        self.overlay.start()
        self.setEnabled(False)

        # Start background loader
        self._loader_thread = ManifestLoaderThread(path)
        self._loader_thread.finished.connect(self._on_manifest_loaded)
        self._loader_thread.failed.connect(self._on_manifest_failed)
        self._loader_thread.start()

    def _on_manifest_loaded(self, data: dict) -> None:
        """Handle successfully loaded manifest."""
        self.current_run_data = data

        # fast populate basic UI (without loading image data)
        self._populate_timeline_quick()

        # start thumbnail generation in background for all steps
        steps = self.current_run_data.get("steps", []) or []
        self._start_thumbnail_thread(steps)

        self.overlay.stop()
        self.setEnabled(True)
        steps_count = len(steps)
        self.status.showMessage(f"Loaded {steps_count} steps from run")

    def _on_manifest_failed(self, error_msg: str) -> None:
        """Handle failure while loading manifest."""
        self.txt_details.setText(f"Failed to load manifest:\n{error_msg}")
        self.overlay.stop()
        self.setEnabled(True)
        self.status.showMessage("Failed to load manifest.")

    def _populate_timeline_quick(self) -> None:
        """Populate timeline and placeholder thumbnails quickly (no pixmap work).

        This avoids expensive image IO in the main thread. Thumbnails are filled
        asynchronously by ThumbnailLoaderThread.
        """
        self.timeline_list.setUpdatesEnabled(False)
        self.thumb_list.setUpdatesEnabled(False)

        self.timeline_list.clear()
        self.thumb_list.clear()
        self._thumb_items_by_index.clear()

        self.state_filter.blockSignals(True)  # noqa: FBT003
        self.state_filter.clear()
        self.state_filter.addItem("All states")
        self.state_filter.blockSignals(False)  # noqa: FBT003

        run_id = self.current_run_data.get("run_id", "UNKNOWN_RUN")
        self.run_label.setText(f" {run_id} ")

        metrics = self.current_run_data.get("summary_metrics", {})
        if metrics:
            processed = metrics.get("processed_count", 0)
            success_rate = metrics.get("success_rate", 0.0)
            self.lbl_metrics.setText(
                f"<b>Processed:</b> {processed}  &nbsp;|&nbsp;  <b>Success rate:</b> {success_rate * 100:.1f}%",
            )
            timings_text = json.dumps(metrics.get("step_timings_sec", {}), indent=2)
            self.metrics_details.setPlainText(timings_text)
        else:
            self.lbl_metrics.setText("No summary metrics present.")
            self.metrics_details.setPlainText("")

        steps = self.current_run_data.get("steps", []) or []
        states = sorted({s.get("fsm_state", "UNKNOWN") for s in steps})
        for st in states:
            self.state_filter.addItem(st)

        # Create timeline items and placeholder thumbnail items fast
        for step in steps:
            ts = self.format_ts(step.get("timestamp", 0))
            idx = step.get("index", 0)
            state = step.get("fsm_state", "UNKNOWN")
            action = step.get("action", "")

            item = QListWidgetItem(f"[{ts}] {idx:03d} | {state}\n{action}")
            item.setData(Qt.ItemDataRole.UserRole, step)
            self.timeline_list.addItem(item)

            thumb_item = QListWidgetItem(f"{idx:03d} {state}")
            thumb_item.setData(Qt.ItemDataRole.UserRole, step)
            # leave icon empty for now; the background thread will fill it
            self.thumb_list.addItem(thumb_item)
            self._thumb_items_by_index[int(idx)] = thumb_item

        # add potential error item
        error = self.current_run_data.get("error_details")
        if error:
            ts = self.format_ts(error.get("timestamp", 0))
            err_type = error.get("exception_type", "Error")
            item = QListWidgetItem(f"[{ts}] FATAL CRASH | {err_type}")
            item.setData(Qt.ItemDataRole.UserRole, {"is_error": True, **error})
            item.setForeground(QColor(ERROR_COLOR))
            self.timeline_list.addItem(item)

        self.timeline_list.setUpdatesEnabled(True)
        self.thumb_list.setUpdatesEnabled(True)

        if self.timeline_list.count() > 0:
            self.timeline_list.setCurrentRow(0)
            self.update_scrub_range()

    def _start_thumbnail_thread(self, steps: list[dict[str, Any]]) -> None:
        """Start a background thread that produces scaled QImage thumbnails."""
        if self._thumb_thread and self._thumb_thread.isRunning():
            with contextlib.suppress(Exception):
                self._thumb_thread.terminate()

        self._thumb_thread = ThumbnailLoaderThread(
            steps,
            icon_size=self.thumb_list.iconSize(),
        )
        self._thumb_thread.finished.connect(self._on_thumbs_ready)
        self._thumb_thread.start()

    def _on_thumbs_ready(self, results: list[dict[str, Any]]) -> None:
        """Receive scaled QImage results and set icons on the thumbnail items."""
        # iterate and set icons; convert QImage -> QPixmap in main thread
        for info in results:
            idx = int(info.get("index", 0))
            qimg = info.get("qimage")
            item = self._thumb_items_by_index.get(idx)
            if item is None:
                continue
            if qimg is None:
                # optional: set a muted placeholder icon or skip
                continue
            pix = QPixmap.fromImage(qimg)
            if not pix.isNull():
                item.setIcon(QIcon(pix))

    def apply_filters(self) -> None:
        """Filter timeline and thumbnails based on search and state."""
        term = self.search_input.text().strip().lower()
        state_filter = self.state_filter.currentText()
        steps_source = self.current_run_data.get("steps", []) or []
        self.filtered_steps = []

        # quick clear lists (we'll repopulate lightweight items)
        self.timeline_list.setUpdatesEnabled(False)
        self.thumb_list.setUpdatesEnabled(False)
        self.timeline_list.clear()
        self.thumb_list.clear()
        self._thumb_items_by_index.clear()

        filtered_steps: list[dict[str, Any]] = []

        for step in steps_source:
            st = step.get("fsm_state", "")
            action = step.get("action", "")
            text_combo = f"{st} {action}".lower()

            if term and term not in text_combo:
                continue
            if state_filter not in ("All states", st):
                continue

            filtered_steps.append(step)

            ts = self.format_ts(step.get("timestamp", 0))
            idx = step.get("index", 0)
            item = QListWidgetItem(f"[{ts}] {idx:03d} | {st}\n{action}")
            item.setData(Qt.ItemDataRole.UserRole, step)
            self.timeline_list.addItem(item)

            thumb_item = QListWidgetItem(f"{idx:03d} {st}")
            thumb_item.setData(Qt.ItemDataRole.UserRole, step)
            self.thumb_list.addItem(thumb_item)
            self._thumb_items_by_index[int(idx)] = thumb_item

        # error item if present
        error = self.current_run_data.get("error_details")
        if error:
            err_item = QListWidgetItem("FATAL CRASH")
            err_item.setData(Qt.ItemDataRole.UserRole, {"is_error": True, **error})
            err_item.setForeground(QColor(ERROR_COLOR))
            self.timeline_list.addItem(err_item)

        self.timeline_list.setUpdatesEnabled(True)
        self.thumb_list.setUpdatesEnabled(True)

        # regenerate thumbnails only for filtered items (background)
        if filtered_steps:
            self._start_thumbnail_thread(filtered_steps)

        self.update_scrub_range()

    def update_scrub_range(self) -> None:
        """Update the navigation slider range."""
        count = self.timeline_list.count()
        self.scrub.setMinimum(0)
        self.scrub.setMaximum(max(0, count - 1))
        self.scrub.setValue(0)
        self.scrub.setEnabled(count > 0)

    def on_search_changed(self, _text: str) -> None:
        """Handle search input changes with a small delay."""
        QTimer.singleShot(180, self.apply_filters)

    def on_filter_changed(self, _text: str) -> None:
        """Handle changes to the state filter dropdown."""
        self.apply_filters()

    def on_thumb_selected(
        self,
        current: QListWidgetItem,
        _previous: QListWidgetItem,
    ) -> None:
        """Select the corresponding timeline entry."""
        if not current:
            return
        data = current.data(Qt.ItemDataRole.UserRole)
        for i in range(self.timeline_list.count()):
            it = self.timeline_list.item(i)
            if it and it.data(Qt.ItemDataRole.UserRole) == data:
                self.timeline_list.setCurrentRow(i)
                break

    def on_step_selected(
        self,
        current: QListWidgetItem,
        _previous: QListWidgetItem,
    ) -> None:
        """Update the viewer with details of the selected step."""
        if not current:
            return
        data = current.data(Qt.ItemDataRole.UserRole)
        idx = data.get("index", None)
        if idx is not None:
            self.lbl_selected.setText(f"Step {idx}")
            self.scrub.setValue(self.timeline_list.row(current))
            self.status.showMessage(f"Selected step {idx}")

        if data.get("is_error"):
            self.image_display.load_image(data.get("crash_screenshot", ""))
            html = f"""
            <span style='color:{ERROR_COLOR}'><b>CRASH DETECTED</b></span><br>
            <b>Type:</b> {data.get("exception_type")}<br>
            <b>Message:</b> {data.get("message")}<br>
            <b>Timestamp:</b> {self.format_ts(data.get("timestamp"))}
            """
            self.txt_details.setHtml(html)
            return

        self.image_display.load_image(data.get("screenshot_path", ""))
        context_json = json.dumps(data.get("context", {}), indent=2)
        html = f"""
        <span style='color:{ACCENT_COLOR}'><b>STATE: {data.get("fsm_state")}</b></span><br>
        <b>Action:</b> {data.get("action")}<br>
        <b>Timestamp:</b> {self.format_ts(data.get("timestamp"))}<br>
        <b>Context:</b><pre>{context_json}</pre>
        """
        self.txt_details.setHtml(html)

    def navigate_step(self, delta: int) -> None:
        """Navigate through the timeline by a delta offset."""
        current_row = self.timeline_list.currentRow()
        if current_row < 0 and self.timeline_list.count() > 0:
            current_row = 0
        target = max(0, min(self.timeline_list.count() - 1, current_row + delta))
        if target >= 0:
            self.timeline_list.setCurrentRow(target)

    def open_video(self) -> None:
        """Open the execution video using a path stored in metadata.json."""
        if not self._monitoring_module:
            QMessageBox.information(
                self,
                "Video unavailable",
                "monitoring module not found.",
            )
            return

        # Check if metadata has a video path field
        video_path_str = self.current_run_data.get("execution_video", "")
        if not video_path_str:
            QMessageBox.information(
                self,
                "No video path",
                "No video path specified in metadata.json.",
            )
            return

        video_path = Path(video_path_str)
        if not video_path.exists():
            QMessageBox.information(
                self,
                "Video not found",
                f"Video file not found at:\n{video_path}",
            )
            return

        # Open with system default application
        try:
            if platform.system() == "Windows":
                subprocess.run(["start", str(video_path)], shell=True, check=False)  # noqa: S602, S607
            elif platform.system() == "Darwin":
                subprocess.run(["open", str(video_path)], check=False)  # noqa: S603, S607
            else:  # Linux
                subprocess.run(["xdg-open", str(video_path)], check=False)  # noqa: S603, S607
        except Exception as e:
            QMessageBox.warning(self, "Failed to open video", str(e))

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Ensure overlay covers the whole client area after resize."""
        super().resizeEvent(event)
        if hasattr(self, "overlay") and self.overlay and self.centralWidget():
            self.overlay.setGeometry(self.centralWidget().rect())


def run() -> None:
    """Launch the application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = TelemetryViewer()
    viewer.showMaximized()
    sys.exit(app.exec())
