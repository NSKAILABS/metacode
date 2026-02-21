"""
Step 2: Phase Curve Display Page.

Shows the interpolated phase-vs-dimension curve and design parameters.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from core.meta_data import MetaDataStore, interpolate_phase_data, FDTDEntry


class PhaseCurvePage(QWidget):
    """Displays the phase-dimension relationship and design parameters."""

    def __init__(self, data_store: MetaDataStore, parent=None):
        super().__init__(parent)
        self.data_store = data_store
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(20)

        # ── Plot area ──
        self.figure = Figure(figsize=(6, 4.5), facecolor='#1e2040')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumWidth(480)
        layout.addWidget(self.canvas, stretch=3)

        # ── Parameters panel ──
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)

        header = QLabel("Design Parameters")
        header.setObjectName("section_header")
        params_layout.addWidget(header)

        self.param_labels = {}
        param_names = [
            ("wavelength", "Wavelength"),
            ("shape", "Shape"),
            ("material", "Material"),
            ("height", "Height"),
            ("period", "Period"),
            ("width", "Width (Cross/Fin)"),
            ("fin_length", "Fin Length"),
            ("reference", "Reference"),
        ]

        params_grid = QGridLayout()
        params_grid.setSpacing(8)
        for idx, (key, name) in enumerate(param_names):
            lbl_name = QLabel(f"{name}:")
            lbl_name.setObjectName("param_label")
            lbl_val = QLabel("—")
            lbl_val.setWordWrap(True)
            self.param_labels[key] = lbl_val
            params_grid.addWidget(lbl_name, idx, 0, Qt.AlignmentFlag.AlignRight)
            params_grid.addWidget(lbl_val, idx, 1)

        params_layout.addLayout(params_grid)
        params_layout.addStretch()
        layout.addWidget(params_widget, stretch=2)

    def update_for_wavelength(self, wl: int):
        """Refresh plot and parameters for the given wavelength."""
        entry = self.data_store.get_entry(wl)
        if entry is None:
            return

        # Update parameter labels
        self.param_labels["wavelength"].setText(f"{wl} nm")
        self.param_labels["shape"].setText(entry.shape)
        self.param_labels["material"].setText(entry.material)
        self.param_labels["height"].setText(f"{entry.height} nm")
        self.param_labels["period"].setText(f"{entry.period} nm")

        if entry.cross_width:
            self.param_labels["width"].setText(f"{entry.cross_width} nm")
        elif entry.fin_width:
            self.param_labels["width"].setText(f"{entry.fin_width} nm")
        else:
            self.param_labels["width"].setText("N/A")

        self.param_labels["fin_length"].setText(
            f"{entry.fin_length} nm" if entry.fin_length else "N/A"
        )
        self.param_labels["reference"].setText(entry.reference or "—")

        # Plot phase curve
        dim_fine, ph_fine, _ = interpolate_phase_data(entry.dimensions, entry.phases)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1a1b2e')
        self.figure.patch.set_facecolor('#1e2040')

        # Plot interpolated curve
        ax.plot(dim_fine, ph_fine, '-', color='#5090e0', linewidth=2, label='Interpolated')
        # Plot data points
        ax.plot(entry.dimensions, entry.phases, 'o', color='#ff8060',
                markersize=7, label='FDTD Data', zorder=5)

        if entry.shape == 'Cylinder':
            xlabel = 'Radius (nm)'
        elif entry.shape == 'Cross':
            xlabel = 'Length L (nm)'
        elif entry.shape == 'Fin':
            xlabel = 'Rotation θ (degrees)'
        else:
            xlabel = 'Dimension'

        ax.set_xlabel(xlabel, color='#b0b8d0', fontsize=11)
        ax.set_ylabel('Phase (degrees)', color='#b0b8d0', fontsize=11)
        ax.set_title(f'Transmission Phase — {wl} nm {entry.shape}',
                     color='#e0e0e8', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#808898')
        ax.grid(True, alpha=0.2, color='#404580')
        ax.legend(facecolor='#22253e', edgecolor='#404580',
                  labelcolor='#c0c8e0', fontsize=10)

        for spine in ax.spines.values():
            spine.set_color('#303560')

        self.figure.tight_layout()
        self.canvas.draw()