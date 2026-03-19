"""
AutoML Page — MetaOpticsAI GUI
================================

PyQt6 page integrating the Self-RAG + LangGraph metalens AutoML engine.

Features:
  • Natural-language requirements input with example prompts
  • Real-time streaming progress log (QThread worker)
  • Live FOM (Strehl ratio) chart updating each iteration
  • Parameter viewer with collapsible JSON tree
  • Export to GDS / save design buttons
  • Ollama model selector + connection status indicator

Compatible with Python 3.11.9+ and the MetaOpticsAI dark theme.
"""

from __future__ import annotations

import json
import sys
import os
import traceback
from typing import Any

from PyQt6.QtCore import (
    QThread, pyqtSignal, Qt, QTimer, QSize, pyqtSlot
)
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPen, QBrush, QTextCursor, QIcon
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QPlainTextEdit, QLineEdit, QComboBox, QGroupBox,
    QSplitter, QScrollArea, QFrame, QProgressBar, QSizePolicy,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QSpinBox, QDoubleSpinBox,
    QCheckBox, QMessageBox, QFileDialog, QApplication, QGridLayout,
    QSlider, QToolButton
)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (extends MetaOpticsAI dark theme)
# ─────────────────────────────────────────────────────────────────────────────

C_BG         = "#1e1e2e"
C_SURFACE    = "#2a2a3e"
C_SURFACE2   = "#313149"
C_ACCENT     = "#7c3aed"          # primary violet
C_ACCENT2    = "#06b6d4"          # cyan accent
C_SUCCESS    = "#22c55e"
C_WARNING    = "#f59e0b"
C_ERROR      = "#ef4444"
C_TEXT       = "#e2e8f0"
C_TEXT_DIM   = "#94a3b8"
C_BORDER     = "#3f3f5a"

PAGE_STYLE = f"""
QWidget {{ background-color: {C_BG}; color: {C_TEXT}; font-family: 'Segoe UI', sans-serif; }}
QGroupBox {{
    border: 1px solid {C_BORDER}; border-radius: 8px;
    margin-top: 10px; padding: 6px;
    font-weight: 600; color: {C_ACCENT2};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER};
    border-radius: 6px; color: {C_TEXT}; padding: 4px 8px;
    font-size: 13px;
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid {C_ACCENT};
}}
QPushButton {{
    background: {C_ACCENT}; color: #fff; border: none;
    border-radius: 7px; padding: 8px 18px; font-weight: 600; font-size: 13px;
}}
QPushButton:hover  {{ background: #6d28d9; }}
QPushButton:pressed{{ background: #5b21b6; }}
QPushButton:disabled {{ background: #4b4b6b; color: {C_TEXT_DIM}; }}
QPushButton#secondary {{
    background: {C_SURFACE2}; border: 1px solid {C_BORDER};
    color: {C_TEXT};
}}
QPushButton#secondary:hover {{ border-color: {C_ACCENT}; color: {C_ACCENT}; }}
QPushButton#success {{
    background: #166534; border: 1px solid {C_SUCCESS}; color: {C_SUCCESS};
}}
QPushButton#danger {{
    background: #7f1d1d; border: 1px solid {C_ERROR}; color: {C_ERROR};
}}
QTabWidget::pane    {{ border: 1px solid {C_BORDER}; border-radius: 6px; background: {C_SURFACE}; }}
QTabBar::tab        {{ background: {C_SURFACE2}; color: {C_TEXT_DIM}; padding: 6px 16px;
                       border-radius: 6px 6px 0 0; margin-right: 2px; }}
QTabBar::tab:selected{{ background: {C_ACCENT}; color: #fff; }}
QProgressBar {{
    border: 1px solid {C_BORDER}; border-radius: 5px;
    background: {C_SURFACE}; text-align: center; color: {C_TEXT}; height: 16px;
}}
QProgressBar::chunk {{ background: {C_ACCENT}; border-radius: 4px; }}
QScrollBar:vertical {{
    background: {C_SURFACE}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{ background: {C_BORDER}; border-radius: 4px; }}
QTreeWidget {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER};
    border-radius: 6px; alternate-background-color: {C_SURFACE2};
}}
QTreeWidget::item:selected {{ background: {C_ACCENT}; }}
QSplitter::handle {{ background: {C_BORDER}; }}
"""

EXAMPLE_PROMPTS = [
    "Design a TiO₂ metalens for 532 nm green laser, focal length 2 mm, diameter 1 mm",
    "Create a near-IR silicon metalens at 1550 nm, f=5 mm, D=2 mm, high NA",
    "Design a GaN UV metalens at 405 nm, f=0.5 mm, D=0.3 mm for microscopy",
    "Achromatic metalens for visible 450-700 nm, f=3 mm, D=1.5 mm, TiO₂ pillars",
    "Custom: wavelength 633 nm, focal length 1 mm, diameter 0.5 mm, TiO₂",
]


# ─────────────────────────────────────────────────────────────────────────────
# WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────

class AutoMLWorker(QThread):
    """
    Runs the MetalensAutoML pipeline in a background thread.
    Emits signals for real-time GUI updates.
    """

    sig_progress   = pyqtSignal(str)             # progress message
    sig_fom_update = pyqtSignal(float)           # new FOM value
    sig_params     = pyqtSignal(dict)            # design params update
    sig_finished   = pyqtSignal(dict)            # final result
    sig_error      = pyqtSignal(str)             # error message

    def __init__(
        self,
        requirements: str,
        model:        str = "deepseek-r1:7b",
        parent=None,
    ):
        super().__init__(parent)
        self.requirements = requirements
        self.model        = model
        self._running     = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            # Import here to avoid circular imports and slow startup
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from core.automl import MetalensAutoML

            def on_message(msg: str):
                if not self._running:
                    return
                self.sig_progress.emit(msg)

            automl = MetalensAutoML(
                model=self.model,
                callback=on_message,
            )

            # Stream through the pipeline
            for step_output in automl.run_stream(self.requirements):
                if not self._running:
                    break

                # Extract info from each step
                for node_name, node_state in step_output.items():
                    if not isinstance(node_state, dict):
                        continue

                    # Emit FOM if updated
                    fom = node_state.get("current_fom")
                    if fom and isinstance(fom, (int, float)) and fom > 0:
                        self.sig_fom_update.emit(float(fom))

                    # Emit design params if updated
                    params = node_state.get("design_params")
                    if params and isinstance(params, dict) and len(params) > 2:
                        self.sig_params.emit(params)

                    # Emit messages
                    messages = node_state.get("messages", [])
                    if messages:
                        self.sig_progress.emit(messages[-1])

            # Get final result
            result = automl.run(self.requirements)
            if self._running:
                self.sig_finished.emit(result)

        except Exception as e:
            tb = traceback.format_exc()
            self.sig_error.emit(f"{e}\n\n{tb}")


# ─────────────────────────────────────────────────────────────────────────────
# STREHL CHART WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class StrehlChartWidget(QWidget):
    """Simple custom painter widget for FOM history chart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values:  list[float] = []
        self.target:  float       = 0.85
        self.setMinimumSize(200, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(160)

    def add_value(self, v: float):
        self.values.append(v)
        self.update()

    def clear(self):
        self.values.clear()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        W, H = self.width(), self.height()
        pad  = 36

        # Background
        p.fillRect(0, 0, W, H, QColor(C_SURFACE))
        p.setPen(QPen(QColor(C_BORDER), 1))
        p.drawRect(pad, 10, W - pad - 10, H - 28)

        if not self.values:
            p.setPen(QColor(C_TEXT_DIM))
            p.drawText(pad, 10, W - pad - 10, H - 28,
                       Qt.AlignmentFlag.AlignCenter, "No data yet")
            p.end()
            return

        chart_w = W - pad - 10
        chart_h = H - 38

        # Y-axis labels
        p.setPen(QColor(C_TEXT_DIM))
        p.setFont(QFont("Segoe UI", 8))
        for y_val in [0.0, 0.5, 1.0]:
            py = int(10 + chart_h * (1.0 - y_val))
            p.drawText(0, py - 7, pad - 4, 14,
                       Qt.AlignmentFlag.AlignRight, f"{y_val:.1f}")
            p.setPen(QPen(QColor(C_BORDER), 1, Qt.PenStyle.DotLine))
            p.drawLine(pad, py, W - 10, py)
            p.setPen(QColor(C_TEXT_DIM))

        # Target line
        target_y = int(10 + chart_h * (1.0 - self.target))
        p.setPen(QPen(QColor(C_WARNING), 1, Qt.PenStyle.DashLine))
        p.drawLine(pad, target_y, W - 10, target_y)
        p.setPen(QColor(C_WARNING))
        p.drawText(W - 50, target_y - 14, 44, 12,
                   Qt.AlignmentFlag.AlignRight, f"target")

        if len(self.values) < 2:
            # Just a dot
            x = pad + chart_w // 2
            y = int(10 + chart_h * (1.0 - max(0.0, min(1.0, self.values[0]))))
            p.setPen(QPen(QColor(C_ACCENT), 2))
            p.setBrush(QBrush(QColor(C_ACCENT)))
            p.drawEllipse(x - 4, y - 4, 8, 8)
        else:
            n = len(self.values)
            # Draw line
            p.setPen(QPen(QColor(C_ACCENT2), 2))
            pts = []
            for i, v in enumerate(self.values):
                x = pad + int(i * chart_w / max(n - 1, 1))
                y = 10 + int(chart_h * (1.0 - max(0.0, min(1.0, v))))
                pts.append((x, y))
            for i in range(len(pts) - 1):
                p.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])

            # Last point highlight
            lx, ly = pts[-1]
            color = QColor(C_SUCCESS) if self.values[-1] >= self.target else QColor(C_ACCENT)
            p.setPen(QPen(color, 2))
            p.setBrush(QBrush(color))
            p.drawEllipse(lx - 5, ly - 5, 10, 10)

        # X label
        p.setPen(QColor(C_TEXT_DIM))
        p.drawText(pad, H - 16, chart_w, 14,
                   Qt.AlignmentFlag.AlignCenter, "Optimization Iteration")

        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# STATUS DOT
# ─────────────────────────────────────────────────────────────────────────────

class StatusDot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = C_TEXT_DIM
        self.setFixedSize(12, 12)

    def set_color(self, hex_color: str):
        self._color = hex_color
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(self._color)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(1, 1, 10, 10)
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class AutoMLPage(QWidget):
    """
    Full AutoML page for MetaOpticsAI.

    Layout
    ──────
    ┌──────────────────────────────────────────────────────────────┐
    │  Header: title + Ollama model selector + connection status   │
    ├───────────────────────────┬──────────────────────────────────┤
    │  LEFT PANEL               │  RIGHT PANEL                     │
    │  • Requirements input     │  • Progress log                  │
    │  • Example prompts        │  • Strehl chart                  │
    │  • Run / Stop buttons     │  • Tabs: Design | Params | Sim   │
    └───────────────────────────┴──────────────────────────────────┘
    """

    # Emitted when a design is finalized (for integration with main window)
    design_ready = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(PAGE_STYLE)
        self._worker: AutoMLWorker | None = None
        self._result: dict = {}
        self._setup_ui()

    # ─────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────────
        root.addWidget(self._build_header())

        # ── Body Splitter ─────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([420, 700])
        splitter.setHandleWidth(6)
        root.addWidget(splitter, 1)

        # ── Status bar ────────────────────────────────────────────────────
        root.addWidget(self._build_status_bar())

    def _build_header(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)

        icon_label = QLabel("🔮")
        icon_label.setFont(QFont("Segoe UI", 24))

        title = QLabel("MetaLens AutoML Designer")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C_ACCENT}; letter-spacing: 1px;")

        sub = QLabel("Self-RAG + LangGraph · Ollama · Automated Design Optimization")
        sub.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")

        h.addWidget(icon_label)
        h.addSpacing(8)
        vt = QVBoxLayout()
        vt.addWidget(title)
        vt.addWidget(sub)
        h.addLayout(vt)
        h.addStretch()

        # Ollama model selector
        model_lbl = QLabel("Model:")
        model_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "deepseek-r1:7b",
            "deepseek-r1:14b",
            "llama3.1:8b",
            "llama3.2:3b",
            "qwen2.5:7b",
            "mistral:7b",
            "phi4:14b",
        ])
        self.model_combo.setFixedWidth(160)
        self.model_combo.setToolTip("Select Ollama model for AutoML")

        self._ollama_dot   = StatusDot()
        self._ollama_label = QLabel("Checking…")
        self._ollama_label.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")

        h.addWidget(model_lbl)
        h.addWidget(self.model_combo)
        h.addSpacing(8)
        h.addWidget(self._ollama_dot)
        h.addWidget(self._ollama_label)

        # Check Ollama connectivity after short delay
        QTimer.singleShot(800, self._check_ollama)
        return w

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 6, 0)
        v.setSpacing(10)

        # ── Requirements ──────────────────────────────────────────────────
        req_group = QGroupBox("📝 Design Requirements")
        rg_layout = QVBoxLayout(req_group)

        self.req_input = QTextEdit()
        self.req_input.setPlaceholderText(
            "Describe your metalens design in natural language…\n\n"
            "Example:\n"
            "Design a TiO₂ metalens for 532 nm green laser,\n"
            "focal length 2 mm, diameter 1 mm,\n"
            "optimize for maximum Strehl ratio."
        )
        self.req_input.setFixedHeight(120)
        self.req_input.setFont(QFont("Segoe UI", 13))
        rg_layout.addWidget(self.req_input)

        # Example prompts
        ex_lbl = QLabel("Quick examples:")
        ex_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        rg_layout.addWidget(ex_lbl)

        for prompt in EXAMPLE_PROMPTS:
            btn = QPushButton(f"↳  {prompt[:55]}…" if len(prompt) > 55 else f"↳  {prompt}")
            btn.setObjectName("secondary")
            btn.setFixedHeight(28)
            btn.setFont(QFont("Segoe UI", 10))
            btn.setToolTip(prompt)
            btn.clicked.connect(lambda checked, p=prompt: self.req_input.setPlainText(p))
            rg_layout.addWidget(btn)

        v.addWidget(req_group)

        # ── Advanced Settings ─────────────────────────────────────────────
        adv_group = QGroupBox("⚙️ Advanced Settings")
        adv_layout = QGridLayout(adv_group)
        adv_layout.setSpacing(6)

        def row(label, widget, r, tooltip=""):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
            adv_layout.addWidget(lbl, r, 0)
            adv_layout.addWidget(widget, r, 1)
            if tooltip:
                widget.setToolTip(tooltip)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(2, 20)
        self.max_iter_spin.setValue(10)
        row("Max iterations:", self.max_iter_spin, 0, "Maximum optimization iterations")

        self.target_fom_spin = QDoubleSpinBox()
        self.target_fom_spin.setRange(0.3, 0.99)
        self.target_fom_spin.setSingleStep(0.05)
        self.target_fom_spin.setValue(0.85)
        self.target_fom_spin.setDecimals(2)
        row("Target Strehl:", self.target_fom_spin, 1, "Target Strehl ratio (0–1)")

        self.rag_k_spin = QSpinBox()
        self.rag_k_spin.setRange(2, 10)
        self.rag_k_spin.setValue(4)
        row("RAG top-k:", self.rag_k_spin, 2, "Number of knowledge chunks to retrieve")

        self.use_rcwa_check = QCheckBox("Use RCWA simulation")
        self.use_rcwa_check.setChecked(True)
        self.use_rcwa_check.setStyleSheet(f"color: {C_TEXT};")
        adv_layout.addWidget(self.use_rcwa_check, 3, 0, 1, 2)

        v.addWidget(adv_group)

        # ── Action Buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.run_btn = QPushButton("▶  Run AutoML")
        self.run_btn.setFixedHeight(42)
        self.run_btn.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.run_btn.clicked.connect(self._start_automl)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setFixedHeight(42)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_automl)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        v.addLayout(btn_row)

        self.export_btn = QPushButton("💾  Export Design")
        self.export_btn.setObjectName("success")
        self.export_btn.setFixedHeight(36)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_design)
        v.addWidget(self.export_btn)

        v.addStretch()
        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 0, 0, 0)
        v.setSpacing(8)

        # ── Progress Bar ──────────────────────────────────────────────────
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)       # indeterminate initially
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setVisible(False)
        prog_row.addWidget(self.progress_bar)
        v.addLayout(prog_row)

        # ── Strehl Chart ──────────────────────────────────────────────────
        chart_group = QGroupBox("📈 Strehl Ratio — Optimization History")
        cg_layout = QVBoxLayout(chart_group)
        cg_layout.setContentsMargins(8, 8, 8, 4)

        self.strehl_chart = StrehlChartWidget()
        cg_layout.addWidget(self.strehl_chart)

        chart_info_row = QHBoxLayout()
        self.fom_label = QLabel("Current Strehl: —")
        self.fom_label.setStyleSheet(f"color: {C_ACCENT2}; font-size: 13px; font-weight: 600;")
        self.iter_label = QLabel("Iteration: 0")
        self.iter_label.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        chart_info_row.addWidget(self.fom_label)
        chart_info_row.addStretch()
        chart_info_row.addWidget(self.iter_label)
        cg_layout.addLayout(chart_info_row)

        v.addWidget(chart_group)

        # ── Tabs: Log | Design Params | Simulation ────────────────────────
        self.tabs = QTabWidget()

        # Tab 1 — Progress Log
        log_tab = QWidget()
        lt = QVBoxLayout(log_tab)
        lt.setContentsMargins(4, 4, 4, 4)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 11))
        self.log_text.setStyleSheet(
            f"background: {C_SURFACE}; color: {C_TEXT}; "
            f"border: 1px solid {C_BORDER}; border-radius: 6px;"
        )
        lt.addWidget(self.log_text)
        clear_btn = QPushButton("Clear Log")
        clear_btn.setObjectName("secondary")
        clear_btn.setFixedHeight(28)
        clear_btn.clicked.connect(self.log_text.clear)
        lt.addWidget(clear_btn)
        self.tabs.addTab(log_tab, "📋 Progress Log")

        # Tab 2 — Design Parameters
        param_tab = QWidget()
        pt = QVBoxLayout(param_tab)
        pt.setContentsMargins(4, 4, 4, 4)
        self.param_tree = QTreeWidget()
        self.param_tree.setHeaderLabels(["Parameter", "Value", "Unit"])
        self.param_tree.setColumnWidth(0, 200)
        self.param_tree.setColumnWidth(1, 130)
        self.param_tree.setAlternatingRowColors(True)
        pt.addWidget(self.param_tree)
        self.tabs.addTab(param_tab, "⚙️ Design Params")

        # Tab 3 — Simulation Results
        sim_tab = QWidget()
        st = QVBoxLayout(sim_tab)
        st.setContentsMargins(4, 4, 4, 4)
        self.sim_tree = QTreeWidget()
        self.sim_tree.setHeaderLabels(["Metric", "Value"])
        self.sim_tree.setColumnWidth(0, 220)
        self.sim_tree.setAlternatingRowColors(True)
        st.addWidget(self.sim_tree)
        self.tabs.addTab(sim_tab, "🔬 Simulation")

        # Tab 4 — Design Summary
        summary_tab = QWidget()
        sumv = QVBoxLayout(summary_tab)
        sumv.setContentsMargins(4, 4, 4, 4)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Segoe UI", 12))
        self.summary_text.setPlaceholderText(
            "Design summary will appear here after AutoML completes…"
        )
        sumv.addWidget(self.summary_text)
        self.tabs.addTab(summary_tab, "📝 Summary")

        v.addWidget(self.tabs, 1)
        return w

    def _build_status_bar(self) -> QWidget:
        w = QFrame()
        w.setFrameShape(QFrame.Shape.HLine)
        w.setFixedHeight(28)
        w.setStyleSheet(f"background: {C_SURFACE2}; border-top: 1px solid {C_BORDER};")
        h = QHBoxLayout(w)
        h.setContentsMargins(10, 0, 10, 0)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")

        self.status_dot = StatusDot()
        self.status_dot.set_color(C_SUCCESS)

        h.addWidget(self.status_dot)
        h.addWidget(self.status_label)
        h.addStretch()

        self.pipeline_label = QLabel("Pipeline: Self-RAG + LangGraph")
        self.pipeline_label.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        h.addWidget(self.pipeline_label)
        return w

    # ─────────────────────────────────────────────────────────────────────
    # OLLAMA STATUS
    # ─────────────────────────────────────────────────────────────────────

    def _check_ollama(self):
        """Ping Ollama and update status indicator."""
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            self._ollama_dot.set_color(C_SUCCESS)
            self._ollama_label.setText("Ollama: connected")
            self._ollama_label.setStyleSheet(f"color: {C_SUCCESS}; font-size: 11px;")
        except Exception:
            self._ollama_dot.set_color(C_ERROR)
            self._ollama_label.setText("Ollama: offline")
            self._ollama_label.setStyleSheet(f"color: {C_ERROR}; font-size: 11px;")

    # ─────────────────────────────────────────────────────────────────────
    # AUTOML CONTROL
    # ─────────────────────────────────────────────────────────────────────

    def _start_automl(self):
        req = self.req_input.toPlainText().strip()
        if not req:
            QMessageBox.warning(self, "Input Required",
                                "Please enter design requirements before running AutoML.")
            return

        # Reset UI
        self.log_text.clear()
        self.strehl_chart.clear()
        self.param_tree.clear()
        self.sim_tree.clear()
        self.summary_text.clear()
        self._iter_count = 0

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_dot.set_color(C_ACCENT)
        self.status_label.setText("AutoML running…")

        model = self.model_combo.currentText()
        self._worker = AutoMLWorker(req, model=model, parent=self)
        self._worker.sig_progress.connect(self._on_progress)
        self._worker.sig_fom_update.connect(self._on_fom)
        self._worker.sig_params.connect(self._on_params)
        self._worker.sig_finished.connect(self._on_finished)
        self._worker.sig_error.connect(self._on_error)
        self._worker.start()

        self._append_log("🚀 AutoML pipeline started", color=C_ACCENT2)
        self._append_log(f"   Model: {model}", color=C_TEXT_DIM)
        self._append_log(f"   Requirements: {req[:80]}…" if len(req) > 80 else f"   Requirements: {req}",
                         color=C_TEXT_DIM)

    def _stop_automl(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.quit()
            self._worker.wait(3000)
        self._reset_run_state()
        self.status_label.setText("Stopped by user")
        self._append_log("⏹ Pipeline stopped by user", color=C_WARNING)

    def _reset_run_state(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

    # ─────────────────────────────────────────────────────────────────────
    # SIGNAL HANDLERS
    # ─────────────────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_progress(self, msg: str):
        # Color-code by emoji prefix
        color = C_TEXT
        if msg.startswith("✅") or msg.startswith("🏁"):
            color = C_SUCCESS
        elif msg.startswith("❌"):
            color = C_ERROR
        elif msg.startswith("⚠"):
            color = C_WARNING
        elif msg.startswith("🧠") or msg.startswith("📊"):
            color = C_ACCENT2
        self._append_log(msg, color=color)

    @pyqtSlot(float)
    def _on_fom(self, fom: float):
        self._iter_count = getattr(self, "_iter_count", 0) + 1
        self.strehl_chart.add_value(fom)
        target = self.target_fom_spin.value()

        color = C_SUCCESS if fom >= target else (C_WARNING if fom >= 0.5 else C_ERROR)
        self.fom_label.setText(f"Current Strehl: {fom:.4f}")
        self.fom_label.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: 600;")
        self.iter_label.setText(f"Iteration: {self._iter_count}")

    @pyqtSlot(dict)
    def _on_params(self, params: dict):
        self._populate_param_tree(params)
        self.tabs.setCurrentIndex(1)   # switch to Params tab

    @pyqtSlot(dict)
    def _on_finished(self, result: dict):
        self._result = result
        self._reset_run_state()

        status = result.get("status", "unknown")
        fom_hist = result.get("fom_history", [])
        best_fom = max(fom_hist) if fom_hist else 0.0

        if status == "success":
            self.status_dot.set_color(C_SUCCESS)
            self.status_label.setText(f"✅ Design complete — Strehl = {best_fom:.4f}")
        elif status in ("converged", "max_iters"):
            self.status_dot.set_color(C_WARNING)
            self.status_label.setText(f"⚠ Converged — Best Strehl = {best_fom:.4f}")
        else:
            self.status_dot.set_color(C_TEXT_DIM)
            self.status_label.setText(f"Finished — Status: {status}")

        # Populate UI
        self._populate_param_tree(result.get("final_design", {}))
        self._populate_sim_tree(result.get("simulation_results", {}))
        self.summary_text.setPlainText(result.get("design_summary", "No summary generated."))
        self.export_btn.setEnabled(bool(result.get("final_design")))

        self._append_log("─" * 56, color=C_BORDER)
        self._append_log(f"🏁 AutoML complete — Status: {status}", color=C_SUCCESS)
        self._append_log(f"   Best Strehl: {best_fom:.4f}", color=C_ACCENT2)
        self.tabs.setCurrentIndex(3)   # show summary tab
        self.design_ready.emit(result.get("final_design", {}))

    @pyqtSlot(str)
    def _on_error(self, error_msg: str):
        self._reset_run_state()
        self.status_dot.set_color(C_ERROR)
        self.status_label.setText("Pipeline error")
        self._append_log(f"❌ Error: {error_msg[:300]}", color=C_ERROR)
        QMessageBox.critical(self, "AutoML Error",
                             f"The AutoML pipeline encountered an error:\n\n{error_msg[:500]}")

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _append_log(self, msg: str, color: str = C_TEXT):
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

        # Use HTML for coloring
        safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.log_text.appendHtml(f'<span style="color:{color};">{safe}</span>')
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _populate_param_tree(self, params: dict):
        self.param_tree.clear()
        if not params:
            return

        UNITS = {
            "wavelength_nm":    "nm",
            "focal_length_mm":  "mm",
            "diameter_mm":      "mm",
            "pillar_height_nm": "nm",
            "min_diameter_nm":  "nm",
            "max_diameter_nm":  "nm",
            "periodicity_nm":   "nm",
            "numerical_aperture": "",
            "f_number":         "",
        }

        CATEGORIES = {
            "Optical Design": ["wavelength_nm", "focal_length_mm", "diameter_mm",
                               "numerical_aperture", "f_number"],
            "Meta-atom":      ["pillar_material", "pillar_height_nm",
                               "min_diameter_nm", "max_diameter_nm", "periodicity_nm"],
            "Simulation":     ["xy_harmonics", "n_radial_pixels", "substrate_index",
                               "jones_vector"],
            "Other":          [],
        }

        classified = set()
        for cat, keys in CATEGORIES.items():
            cat_item = QTreeWidgetItem([cat, "", ""])
            cat_item.setForeground(0, QColor(C_ACCENT2))
            cat_item.setFont(0, QFont("Segoe UI", 11, QFont.Weight.Bold))
            cat_item.setExpanded(True)

            for key in keys:
                if key not in params:
                    continue
                val = params[key]
                unit = UNITS.get(key, "")
                if isinstance(val, float):
                    val_str = f"{val:.4g}"
                else:
                    val_str = str(val)
                child = QTreeWidgetItem([key, val_str, unit])
                child.setForeground(0, QColor(C_TEXT))
                child.setForeground(1, QColor(C_ACCENT))
                cat_item.addChild(child)
                classified.add(key)

            if cat_item.childCount() > 0:
                self.param_tree.addTopLevelItem(cat_item)

        # Remaining unclassified params
        other_item = QTreeWidgetItem(["Other", "", ""])
        other_item.setForeground(0, QColor(C_ACCENT2))
        other_item.setExpanded(True)
        for k, v in params.items():
            if k not in classified:
                unit = UNITS.get(k, "")
                val_str = f"{v:.4g}" if isinstance(v, float) else str(v)
                child = QTreeWidgetItem([k, val_str, unit])
                child.setForeground(0, QColor(C_TEXT))
                child.setForeground(1, QColor(C_TEXT_DIM))
                other_item.addChild(child)
        if other_item.childCount() > 0:
            self.param_tree.addTopLevelItem(other_item)

    def _populate_sim_tree(self, results: dict):
        self.sim_tree.clear()
        if not results:
            return

        DISPLAY_NAMES = {
            "strehl_ratio":     ("Strehl Ratio",     C_SUCCESS),
            "transmission":     ("Transmission",      C_ACCENT2),
            "phase_range_rad":  ("Phase Range",       C_TEXT),
            "phase_uniformity": ("Phase Uniformity",  C_TEXT),
            "na":               ("Numerical Aperture",C_ACCENT),
            "f_number":         ("F-number",          C_TEXT),
            "engine":           ("Simulation Engine", C_TEXT_DIM),
            "notes":            ("Notes",             C_TEXT_DIM),
        }

        for key, val in results.items():
            display, color = DISPLAY_NAMES.get(key, (key, C_TEXT))
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            item = QTreeWidgetItem([display, val_str])
            item.setForeground(0, QColor(C_TEXT))
            item.setForeground(1, QColor(color))
            self.sim_tree.addTopLevelItem(item)

    # ─────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────

    def _export_design(self):
        if not self._result:
            return

        path, flt = QFileDialog.getSaveFileName(
            self,
            "Export Metalens Design",
            "metalens_design.json",
            "JSON Design File (*.json);;All Files (*)",
        )
        if not path:
            return

        export_data = {
            "final_design":       self._result.get("final_design", {}),
            "simulation_results": self._result.get("simulation_results", {}),
            "design_summary":     self._result.get("design_summary", ""),
            "fom_history":        self._result.get("fom_history", []),
            "param_history":      self._result.get("param_history", []),
            "status":             self._result.get("status", ""),
        }

        try:
            with open(path, "w") as f:
                json.dump(export_data, f, indent=2)
            self._append_log(f"💾 Design exported to: {path}", color=C_SUCCESS)
            self.status_label.setText(f"Exported to {os.path.basename(path)}")
            QMessageBox.information(self, "Export Successful",
                                    f"Design exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = QWidget()
    window.setWindowTitle("MetaOpticsAI — AutoML Page Preview")
    window.resize(1280, 820)
    window.setStyleSheet(f"background: {C_BG};")

    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    page = AutoMLPage()
    layout.addWidget(page)

    window.show()
    sys.exit(app.exec())