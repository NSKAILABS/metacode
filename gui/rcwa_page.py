"""
Step 1: RCWA Unit Cell Simulation Page.

Implements the MAXIM-style parametric sweep interface for computing
transmission/reflection coefficients of metasurface unit cells.
This is the first step because the phase-vs-dimension data generated
here feeds all downstream design steps.

Layout: Scrollable config panel (left) | Tabbed results (right)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QComboBox, QDoubleSpinBox, QSpinBox,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QFileDialog, QScrollArea, QFrame, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt, QThread
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from core.rcwa_engine import RCWAEngine, RCWAConfig, RCWAStructure, parametric_sweep, RCWAResult
from core.design import DesignState


class SweepWorker(QThread):
    """Background thread for parametric sweep."""
    progress = pyqtSignal(float)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, config, sweep_param, sweep_values):
        super().__init__()
        self.config = config
        self.sweep_param = sweep_param
        self.sweep_values = sweep_values

    def run(self):
        try:
            results = parametric_sweep(
                self.config, self.sweep_param, self.sweep_values,
                progress_callback=lambda p: self.progress.emit(p)
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


def _compact_double(val, lo, hi, dec=2, suffix=""):
    """Create a compact QDoubleSpinBox."""
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(val)
    sb.setDecimals(dec)
    sb.setFixedHeight(28)
    if suffix:
        sb.setSuffix(suffix)
    return sb


def _compact_int(val, lo, hi):
    """Create a compact QSpinBox."""
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(val)
    sb.setFixedHeight(28)
    return sb


class RCWAPage(QWidget):
    """
    RCWA unit cell simulation interface with parametric sweep.
    Now Step 1 — generates the phase library used by all downstream steps.
    """

    # Signal emitted when sweep data is available for use in later steps
    sweep_data_ready = pyqtSignal()

    def __init__(self, design: DesignState, parent=None):
        super().__init__(parent)
        self.design = design
        self._results = []
        self._sweep_values = []
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(10)

        # ══════════════════════════════════════════════════════════════
        # LEFT: Config panel inside a scroll area
        # ══════════════════════════════════════════════════════════════
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { width: 6px; }
        """)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(380)

        config_widget = QWidget()
        cl = QVBoxLayout(config_widget)
        cl.setContentsMargins(4, 4, 8, 4)
        cl.setSpacing(6)

        # ── Section 1: Unit Cell ──
        sec1 = QGroupBox("Unit Cell Structure")
        sec1.setStyleSheet("QGroupBox { padding-top: 14px; margin-top: 6px; }")
        g1 = QGridLayout()
        g1.setContentsMargins(8, 4, 8, 6)
        g1.setVerticalSpacing(4)
        g1.setHorizontalSpacing(8)

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["circle", "rectangle", "film"])
        self.shape_combo.setFixedHeight(28)
        g1.addWidget(QLabel("Shape"), 0, 0)
        g1.addWidget(self.shape_combo, 0, 1)

        self.n_struct_real = _compact_double(3.5, 0.1, 10.0, 3)
        g1.addWidget(QLabel("n (real)"), 1, 0)
        g1.addWidget(self.n_struct_real, 1, 1)

        self.n_struct_imag = _compact_double(0.0, 0.0, 5.0, 4)
        g1.addWidget(QLabel("k (imag)"), 2, 0)
        g1.addWidget(self.n_struct_imag, 2, 1)

        self.n_surr = _compact_double(1.0, 0.1, 5.0, 3)
        g1.addWidget(QLabel("n surround"), 3, 0)
        g1.addWidget(self.n_surr, 3, 1)

        self.height_spin = _compact_double(220, 1, 50000, 1, " nm")
        g1.addWidget(QLabel("Height"), 4, 0)
        g1.addWidget(self.height_spin, 4, 1)

        sec1.setLayout(g1)
        cl.addWidget(sec1)

        # ── Section 2: Simulation ──
        sec2 = QGroupBox("Simulation")
        sec2.setStyleSheet("QGroupBox { padding-top: 14px; margin-top: 2px; }")
        g2 = QGridLayout()
        g2.setContentsMargins(8, 4, 8, 6)
        g2.setVerticalSpacing(4)
        g2.setHorizontalSpacing(8)

        self.wl_spin = _compact_double(532, 100, 100000, 1, " nm")
        g2.addWidget(QLabel("Wavelength"), 0, 0)
        g2.addWidget(self.wl_spin, 0, 1)

        self.px_spin = _compact_double(350, 10, 100000, 1, " nm")
        g2.addWidget(QLabel("Period X"), 1, 0)
        g2.addWidget(self.px_spin, 1, 1)

        self.py_spin = _compact_double(350, 10, 100000, 1, " nm")
        g2.addWidget(QLabel("Period Y"), 2, 0)
        g2.addWidget(self.py_spin, 2, 1)

        # Truncation M and N side by side
        trunc_row = QHBoxLayout()
        trunc_row.setSpacing(6)
        trunc_row.addWidget(QLabel("M"))
        self.M_spin = _compact_int(5, 1, 30)
        trunc_row.addWidget(self.M_spin)
        trunc_row.addWidget(QLabel("N"))
        self.N_spin = _compact_int(5, 1, 30)
        trunc_row.addWidget(self.N_spin)
        g2.addWidget(QLabel("Truncation"), 3, 0)
        trunc_container = QWidget()
        trunc_container.setLayout(trunc_row)
        g2.addWidget(trunc_container, 3, 1)

        # Input/output n side by side
        n_io_row = QHBoxLayout()
        n_io_row.setSpacing(6)
        n_io_row.addWidget(QLabel("in"))
        self.n_in_spin = _compact_double(1.46, 0.1, 5.0, 3)
        n_io_row.addWidget(self.n_in_spin)
        n_io_row.addWidget(QLabel("out"))
        self.n_out_spin = _compact_double(1.0, 0.1, 5.0, 3)
        n_io_row.addWidget(self.n_out_spin)
        g2.addWidget(QLabel("Medium n"), 4, 0)
        n_io_container = QWidget()
        n_io_container.setLayout(n_io_row)
        g2.addWidget(n_io_container, 4, 1)

        sec2.setLayout(g2)
        cl.addWidget(sec2)

        # ── Section 3: Parametric Sweep ──
        sec3 = QGroupBox("Parametric Sweep")
        sec3.setStyleSheet("QGroupBox { padding-top: 14px; margin-top: 2px; }")
        g3 = QGridLayout()
        g3.setContentsMargins(8, 4, 8, 6)
        g3.setVerticalSpacing(4)
        g3.setHorizontalSpacing(8)

        self.sweep_combo = QComboBox()
        self.sweep_combo.addItems(["radius", "diameter", "wx", "wy", "height", "wavelength"])
        self.sweep_combo.setFixedHeight(28)
        g3.addWidget(QLabel("Parameter"), 0, 0)
        g3.addWidget(self.sweep_combo, 0, 1)

        # Start / End side by side
        range_row = QHBoxLayout()
        range_row.setSpacing(4)
        self.sweep_start = _compact_double(40, 1, 100000, 1)
        self.sweep_end = _compact_double(140, 1, 100000, 1)
        range_row.addWidget(self.sweep_start)
        range_row.addWidget(QLabel("→"))
        range_row.addWidget(self.sweep_end)
        g3.addWidget(QLabel("Range (nm)"), 1, 0)
        range_container = QWidget()
        range_container.setLayout(range_row)
        g3.addWidget(range_container, 1, 1)

        self.sweep_steps = _compact_int(20, 2, 200)
        g3.addWidget(QLabel("Steps"), 2, 0)
        g3.addWidget(self.sweep_steps, 2, 1)

        sec3.setLayout(g3)
        cl.addWidget(sec3)

        # ── Run button ──
        self.run_btn = QPushButton("▶  Run RCWA Sweep")
        self.run_btn.setObjectName("primary")
        self.run_btn.setFixedHeight(38)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #3060c0;
                border: 1px solid #4080e0;
                border-radius: 8px;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #3870d8; }
            QPushButton:disabled { background-color: #1e2040; color: #505570; }
        """)
        self.run_btn.clicked.connect(self._run_sweep)
        cl.addWidget(self.run_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none; border-radius: 4px;
                background-color: #22253e; text-align: center;
                font-size: 9px; color: #8890a8;
            }
            QProgressBar::chunk { background-color: #3070d0; border-radius: 4px; }
        """)
        cl.addWidget(self.progress_bar)

        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        self.status_label.setWordWrap(True)
        cl.addWidget(self.status_label)

        cl.addStretch()

        scroll.setWidget(config_widget)
        root.addWidget(scroll)

        # ══════════════════════════════════════════════════════════════
        # RIGHT: Results (tabs + export)
        # ══════════════════════════════════════════════════════════════
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        self.tabs = QTabWidget()

        # Tab 1: Transmittance + Phase
        self.trans_fig = Figure(figsize=(7, 4), facecolor='#1e2040')
        self.trans_canvas = FigureCanvas(self.trans_fig)
        self.tabs.addTab(self.trans_canvas, "Transmittance  Phase")

        # Tab 2: Diffraction Efficiency
        self.diff_fig = Figure(figsize=(7, 4), facecolor='#1e2040')
        self.diff_canvas = FigureCanvas(self.diff_fig)
        self.tabs.addTab(self.diff_canvas, "Diffraction Efficiency")

        # Tab 3: Data Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Param (nm)", "Transmittance", "Reflectance", "T Phase (rad)", "R Phase (rad)"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.results_table, "Data Table")

        rl.addWidget(self.tabs, stretch=1)

        # Export row
        export_row = QHBoxLayout()
        export_row.setSpacing(8)

        export_btn = QPushButton("Export CSV")
        export_btn.setObjectName("accent")
        export_btn.setFixedHeight(32)
        export_btn.clicked.connect(self._export_csv)
        export_row.addWidget(export_btn)

        use_btn = QPushButton("Use as Phase Library →")
        use_btn.setObjectName("primary")
        use_btn.setFixedHeight(32)
        use_btn.setToolTip("Send sweep results to the wavelength/phase curve steps")
        use_btn.clicked.connect(self._use_as_library)
        export_row.addWidget(use_btn)

        export_row.addStretch()
        rl.addLayout(export_row)

        root.addWidget(right, stretch=1)

    # ─── Plotting helpers ────────────────────────────────────────────

    def _style_ax(self, ax):
        ax.set_facecolor('#1a1b2e')
        ax.tick_params(colors='#808898', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#303560')
        ax.grid(True, alpha=0.15, color='#404580')

    # ─── Config builder ──────────────────────────────────────────────

    def _build_config(self) -> RCWAConfig:
        n_struct = complex(self.n_struct_real.value(), self.n_struct_imag.value())
        layer = RCWAStructure(
            shape=self.shape_combo.currentText(),
            n_struct=n_struct,
            n_surround=complex(self.n_surr.value()),
            height=self.height_spin.value(),
            radius=80,
            wx=100, wy=100,
        )
        config = RCWAConfig(
            wavelength=self.wl_spin.value(),
            Tx=self.px_spin.value(),
            Ty=self.py_spin.value(),
            M=self.M_spin.value(),
            N=self.N_spin.value(),
            n_input=complex(self.n_in_spin.value()),
            n_output=complex(self.n_out_spin.value()),
            layers=[layer],
        )
        return config

    # ─── Sweep execution ─────────────────────────────────────────────

    def _run_sweep(self):
        config = self._build_config()
        sweep_values = np.linspace(
            self.sweep_start.value(), self.sweep_end.value(),
            self.sweep_steps.value()
        ).tolist()
        self._sweep_values = sweep_values

        self.progress_bar.setValue(0)
        self.status_label.setText("Running RCWA sweep...")
        self.status_label.setStyleSheet("font-size: 11px; color: #c0c8e0;")
        self.run_btn.setEnabled(False)

        self._worker = SweepWorker(config, self.sweep_combo.currentText(), sweep_values)
        self._worker.progress.connect(lambda v: self.progress_bar.setValue(int(v)))
        self._worker.finished.connect(self._on_sweep_done)
        self._worker.error.connect(self._on_sweep_error)
        self._worker.start()

    def _on_sweep_done(self, results: list):
        self._results = results
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"✓ Sweep complete — {len(results)} points computed")
        self.status_label.setStyleSheet("font-size: 11px; color: #6ecf6e; font-weight: bold;")
        self._plot_results()

    def _on_sweep_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"✗ Error: {msg}")
        self.status_label.setStyleSheet("font-size: 11px; color: #ff6b6b;")

    # ─── Plot results ────────────────────────────────────────────────

    def _plot_results(self):
        if not self._results:
            return

        vals = self._sweep_values
        T = [r.transmittance for r in self._results]
        R = [r.reflectance for r in self._results]
        T_ph = [r.T_phase for r in self._results]
        R_ph = [r.R_phase for r in self._results]
        param_name = self.sweep_combo.currentText()

        # ── Tab 1: T/R + Phase ──
        self.trans_fig.clear()
        ax1 = self.trans_fig.add_subplot(121)
        ax2 = self.trans_fig.add_subplot(122)
        self._style_ax(ax1)
        self._style_ax(ax2)

        ax1.plot(vals, T, 'o-', color='#5090e0', markersize=3, linewidth=1.5, label='T')
        ax1.plot(vals, R, 's-', color='#ff8060', markersize=3, linewidth=1.5, label='R')
        ax1.set_xlabel(f'{param_name} (nm)', color='#b0b8d0', fontsize=10)
        ax1.set_ylabel('Coefficient', color='#b0b8d0', fontsize=10)
        ax1.set_title('Transmittance & Reflectance', color='#e0e0e8', fontsize=11)
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(facecolor='#22253e', edgecolor='#404580', labelcolor='#c0c8e0', fontsize=9)

        ax2.plot(vals, T_ph, 'o-', color='#5090e0', markersize=3, linewidth=1.5, label='T phase')
        ax2.plot(vals, R_ph, 's-', color='#ff8060', markersize=3, linewidth=1.5, label='R phase')
        ax2.set_xlabel(f'{param_name} (nm)', color='#b0b8d0', fontsize=10)
        ax2.set_ylabel('Phase (rad)', color='#b0b8d0', fontsize=10)
        ax2.set_title('Transmission & Reflection Phase', color='#e0e0e8', fontsize=11)
        ax2.legend(facecolor='#22253e', edgecolor='#404580', labelcolor='#c0c8e0', fontsize=9)

        self.trans_fig.patch.set_facecolor('#1e2040')
        self.trans_fig.tight_layout(pad=1.5)
        self.trans_canvas.draw()

        # ── Tab 2: Diffraction ──
        self.diff_fig.clear()
        ax_d = self.diff_fig.add_subplot(111)
        self._style_ax(ax_d)

        eff_T0 = [r.diff_eff_T.get((0, 0), 0) for r in self._results]
        eff_R0 = [r.diff_eff_R.get((0, 0), 0) for r in self._results]
        ax_d.plot(vals, eff_T0, 'o-', color='#5090e0', markersize=3, label='T (0,0)')
        ax_d.plot(vals, eff_R0, 's-', color='#ff8060', markersize=3, label='R (0,0)')
        ax_d.set_xlabel(f'{param_name} (nm)', color='#b0b8d0', fontsize=10)
        ax_d.set_ylabel('Diffraction Efficiency', color='#b0b8d0', fontsize=10)
        ax_d.set_title('Zeroth-Order Diffraction Efficiency', color='#e0e0e8', fontsize=11)
        ax_d.legend(facecolor='#22253e', edgecolor='#404580', labelcolor='#c0c8e0', fontsize=9)

        self.diff_fig.patch.set_facecolor('#1e2040')
        self.diff_fig.tight_layout(pad=1.5)
        self.diff_canvas.draw()

        # ── Tab 3: Table ──
        self.results_table.setRowCount(len(self._results))
        for row, (v, r) in enumerate(zip(vals, self._results)):
            self.results_table.setItem(row, 0, QTableWidgetItem(f"{v:.2f}"))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{r.transmittance:.6f}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{r.reflectance:.6f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{r.T_phase:.4f}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{r.R_phase:.4f}"))

    # ─── Export / Use ────────────────────────────────────────────────

    def _export_csv(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "rcwa_results.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, 'w') as f:
            f.write("param_value_nm,transmittance,reflectance,T_phase_rad,R_phase_rad\n")
            for v, r in zip(self._sweep_values, self._results):
                f.write(f"{v},{r.transmittance},{r.reflectance},{r.T_phase},{r.R_phase}\n")

    def _use_as_library(self):
        """Push sweep data into the DesignState for downstream steps."""
        if not self._results or not self._sweep_values:
            return
        # Store dimensions (sweep values) and phases (transmission phase)
        self.design.dimensions = [float(v) for v in self._sweep_values]
        self.design.phases = [float(np.degrees(r.T_phase)) for r in self._results]
        self.design.transmissions = [float(r.transmittance) for r in self._results]
        self.design.wavelength = int(self.wl_spin.value())
        self.design.height = self.height_spin.value()
        self.design.period = self.px_spin.value()
        shape_map = {"circle": "Cylinder", "rectangle": "Cross", "film": "Film"}
        self.design.shape = shape_map.get(self.shape_combo.currentText(), "Cylinder")
        self.sweep_data_ready.emit()