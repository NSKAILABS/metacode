"""
Step 6: RCWA Electromagnetic Simulation Page.
Implements the MAXIM-style parametric sweep interface for computing
transmission/reflection coefficients of metasurface unit cells.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QComboBox, QDoubleSpinBox, QSpinBox,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QFileDialog
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


class RCWAPage(QWidget):
    """RCWA simulation interface with parametric sweep capability."""

    def __init__(self, design: DesignState, parent=None):
        super().__init__(parent)
        self.design = design
        self._results = []
        self._sweep_values = []
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(12)

        # Left: Configuration panel
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setSpacing(8)

        # Structure config
        struct_group = QGroupBox("Unit Cell Structure")
        sg = QGridLayout()
        sg.setSpacing(6)

        sg.addWidget(QLabel("Shape:"), 0, 0)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["circle", "rectangle", "film"])
        sg.addWidget(self.shape_combo, 0, 1)

        sg.addWidget(QLabel("Material n:"), 1, 0)
        self.n_struct_real = QDoubleSpinBox()
        self.n_struct_real.setRange(0.1, 10.0)
        self.n_struct_real.setValue(3.5)
        self.n_struct_real.setDecimals(3)
        sg.addWidget(self.n_struct_real, 1, 1)

        sg.addWidget(QLabel("Material k:"), 2, 0)
        self.n_struct_imag = QDoubleSpinBox()
        self.n_struct_imag.setRange(0.0, 5.0)
        self.n_struct_imag.setValue(0.0)
        self.n_struct_imag.setDecimals(4)
        sg.addWidget(self.n_struct_imag, 2, 1)

        sg.addWidget(QLabel("Surround n:"), 3, 0)
        self.n_surr = QDoubleSpinBox()
        self.n_surr.setRange(0.1, 5.0)
        self.n_surr.setValue(1.0)
        self.n_surr.setDecimals(3)
        sg.addWidget(self.n_surr, 3, 1)

        sg.addWidget(QLabel("Height (nm):"), 4, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1, 50000)
        self.height_spin.setValue(220)
        sg.addWidget(self.height_spin, 4, 1)

        struct_group.setLayout(sg)
        ll.addWidget(struct_group)

        # Simulation config
        sim_group = QGroupBox("Simulation Parameters")
        simgl = QGridLayout()
        simgl.setSpacing(6)

        simgl.addWidget(QLabel("Wavelength (nm):"), 0, 0)
        self.wl_spin = QDoubleSpinBox()
        self.wl_spin.setRange(100, 100000)
        self.wl_spin.setValue(532)
        simgl.addWidget(self.wl_spin, 0, 1)

        simgl.addWidget(QLabel("Period X (nm):"), 1, 0)
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(10, 100000)
        self.px_spin.setValue(350)
        simgl.addWidget(self.px_spin, 1, 1)

        simgl.addWidget(QLabel("Period Y (nm):"), 2, 0)
        self.py_spin = QDoubleSpinBox()
        self.py_spin.setRange(10, 100000)
        self.py_spin.setValue(350)
        simgl.addWidget(self.py_spin, 2, 1)

        simgl.addWidget(QLabel("Truncation M:"), 3, 0)
        self.M_spin = QSpinBox()
        self.M_spin.setRange(1, 30)
        self.M_spin.setValue(5)
        simgl.addWidget(self.M_spin, 3, 1)

        simgl.addWidget(QLabel("Truncation N:"), 4, 0)
        self.N_spin = QSpinBox()
        self.N_spin.setRange(1, 30)
        self.N_spin.setValue(5)
        simgl.addWidget(self.N_spin, 4, 1)

        simgl.addWidget(QLabel("Input n (substrate):"), 5, 0)
        self.n_in_spin = QDoubleSpinBox()
        self.n_in_spin.setRange(0.1, 5.0)
        self.n_in_spin.setValue(1.46)
        self.n_in_spin.setDecimals(3)
        simgl.addWidget(self.n_in_spin, 5, 1)

        simgl.addWidget(QLabel("Output n:"), 6, 0)
        self.n_out_spin = QDoubleSpinBox()
        self.n_out_spin.setRange(0.1, 5.0)
        self.n_out_spin.setValue(1.0)
        self.n_out_spin.setDecimals(3)
        simgl.addWidget(self.n_out_spin, 6, 1)

        sim_group.setLayout(simgl)
        ll.addWidget(sim_group)

        # Sweep config
        sweep_group = QGroupBox("Parametric Sweep")
        swgl = QGridLayout()
        swgl.setSpacing(6)

        swgl.addWidget(QLabel("Sweep param:"), 0, 0)
        self.sweep_combo = QComboBox()
        self.sweep_combo.addItems(["radius", "diameter", "wx", "wy", "height", "wavelength"])
        swgl.addWidget(self.sweep_combo, 0, 1)

        swgl.addWidget(QLabel("Start (nm):"), 1, 0)
        self.sweep_start = QDoubleSpinBox()
        self.sweep_start.setRange(1, 100000)
        self.sweep_start.setValue(40)
        swgl.addWidget(self.sweep_start, 1, 1)

        swgl.addWidget(QLabel("End (nm):"), 2, 0)
        self.sweep_end = QDoubleSpinBox()
        self.sweep_end.setRange(1, 100000)
        self.sweep_end.setValue(140)
        swgl.addWidget(self.sweep_end, 2, 1)

        swgl.addWidget(QLabel("Steps:"), 3, 0)
        self.sweep_steps = QSpinBox()
        self.sweep_steps.setRange(2, 200)
        self.sweep_steps.setValue(20)
        swgl.addWidget(self.sweep_steps, 3, 1)

        sweep_group.setLayout(swgl)
        ll.addWidget(sweep_group)

        # Run button + progress
        run_btn = QPushButton("Run RCWA Sweep")
        run_btn.setObjectName("primary")
        run_btn.clicked.connect(self._run_sweep)
        ll.addWidget(run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        ll.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        ll.addWidget(self.status_label)
        ll.addStretch()

        layout.addWidget(left, stretch=2)

        # Right: Results
        right = QWidget()
        rl = QVBoxLayout(right)

        self.tabs = QTabWidget()

        # Transmittance plot
        self.trans_fig = Figure(figsize=(6, 4), facecolor='#1e2040')
        self.trans_canvas = FigureCanvas(self.trans_fig)
        self.tabs.addTab(self.trans_canvas, "Transmittance & Phase")

        # Diffraction efficiency
        self.diff_fig = Figure(figsize=(6, 4), facecolor='#1e2040')
        self.diff_canvas = FigureCanvas(self.diff_fig)
        self.tabs.addTab(self.diff_canvas, "Diffraction Efficiency")

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Param Value", "Transmittance", "Reflectance", "T Phase (rad)", "R Phase (rad)"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.results_table, "Data Table")

        rl.addWidget(self.tabs)

        # Export
        export_btn = QPushButton("Export Results (CSV)")
        export_btn.setObjectName("accent")
        export_btn.clicked.connect(self._export_csv)
        rl.addWidget(export_btn)

        layout.addWidget(right, stretch=3)

    def _style_ax(self, ax):
        ax.set_facecolor('#1a1b2e')
        ax.tick_params(colors='#808898', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#303560')
        ax.grid(True, alpha=0.15, color='#404580')

    def _build_config(self) -> RCWAConfig:
        n_struct = complex(self.n_struct_real.value(), self.n_struct_imag.value())
        layer = RCWAStructure(
            shape=self.shape_combo.currentText(),
            n_struct=n_struct,
            n_surround=complex(self.n_surr.value()),
            height=self.height_spin.value(),
            radius=80,  # default, will be swept
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

    def _run_sweep(self):
        config = self._build_config()
        sweep_values = np.linspace(
            self.sweep_start.value(), self.sweep_end.value(),
            self.sweep_steps.value()
        ).tolist()
        self._sweep_values = sweep_values

        self.progress_bar.setValue(0)
        self.status_label.setText("Running RCWA sweep...")

        self._worker = SweepWorker(config, self.sweep_combo.currentText(), sweep_values)
        self._worker.progress.connect(lambda v: self.progress_bar.setValue(int(v)))
        self._worker.finished.connect(self._on_sweep_done)
        self._worker.error.connect(self._on_sweep_error)
        self._worker.start()

    def _on_sweep_done(self, results: list):
        self._results = results
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Sweep complete: {len(results)} points")
        self.status_label.setObjectName("status_ok")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self._plot_results()

    def _on_sweep_error(self, msg: str):
        self.status_label.setText(f"Error: {msg}")
        self.status_label.setObjectName("status_error")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _plot_results(self):
        if not self._results:
            return

        vals = self._sweep_values
        T = [r.transmittance for r in self._results]
        R = [r.reflectance for r in self._results]
        T_ph = [r.T_phase for r in self._results]
        R_ph = [r.R_phase for r in self._results]
        param_name = self.sweep_combo.currentText()

        # Transmittance + Phase plot
        self.trans_fig.clear()
        ax1 = self.trans_fig.add_subplot(121)
        ax2 = self.trans_fig.add_subplot(122)
        self._style_ax(ax1)
        self._style_ax(ax2)

        ax1.plot(vals, T, 'o-', color='#5090e0', markersize=4, linewidth=1.5, label='T')
        ax1.plot(vals, R, 's-', color='#ff8060', markersize=4, linewidth=1.5, label='R')
        ax1.set_xlabel(f'{param_name} (nm)', color='#b0b8d0')
        ax1.set_ylabel('Coefficient', color='#b0b8d0')
        ax1.set_title('Transmittance & Reflectance', color='#e0e0e8', fontsize=11)
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(facecolor='#22253e', edgecolor='#404580', labelcolor='#c0c8e0')

        ax2.plot(vals, T_ph, 'o-', color='#5090e0', markersize=4, linewidth=1.5, label='T phase')
        ax2.plot(vals, R_ph, 's-', color='#ff8060', markersize=4, linewidth=1.5, label='R phase')
        ax2.set_xlabel(f'{param_name} (nm)', color='#b0b8d0')
        ax2.set_ylabel('Phase (rad)', color='#b0b8d0')
        ax2.set_title('Transmission & Reflection Phase', color='#e0e0e8', fontsize=11)
        ax2.legend(facecolor='#22253e', edgecolor='#404580', labelcolor='#c0c8e0')

        self.trans_fig.patch.set_facecolor('#1e2040')
        self.trans_fig.tight_layout()
        self.trans_canvas.draw()

        # Diffraction efficiency
        self.diff_fig.clear()
        ax_d = self.diff_fig.add_subplot(111)
        self._style_ax(ax_d)

        # Plot zeroth-order efficiency
        eff_T0 = [r.diff_eff_T.get((0, 0), 0) for r in self._results]
        eff_R0 = [r.diff_eff_R.get((0, 0), 0) for r in self._results]
        ax_d.plot(vals, eff_T0, 'o-', color='#5090e0', markersize=4, label='T (0,0)')
        ax_d.plot(vals, eff_R0, 's-', color='#ff8060', markersize=4, label='R (0,0)')
        ax_d.set_xlabel(f'{param_name} (nm)', color='#b0b8d0')
        ax_d.set_ylabel('Diffraction Efficiency', color='#b0b8d0')
        ax_d.set_title('Zeroth-Order Diffraction Efficiency', color='#e0e0e8', fontsize=11)
        ax_d.legend(facecolor='#22253e', edgecolor='#404580', labelcolor='#c0c8e0')

        self.diff_fig.patch.set_facecolor('#1e2040')
        self.diff_fig.tight_layout()
        self.diff_canvas.draw()

        # Table
        self.results_table.setRowCount(len(self._results))
        for row, (v, r) in enumerate(zip(vals, self._results)):
            self.results_table.setItem(row, 0, QTableWidgetItem(f"{v:.2f}"))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{r.transmittance:.6f}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{r.reflectance:.6f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{r.T_phase:.4f}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{r.R_phase:.4f}"))

    def _export_csv(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "rcwa_results.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, 'w') as f:
            f.write("param_value,transmittance,reflectance,T_phase_rad,R_phase_rad\n")
            for v, r in zip(self._sweep_values, self._results):
                f.write(f"{v},{r.transmittance},{r.reflectance},{r.T_phase},{r.R_phase}\n")