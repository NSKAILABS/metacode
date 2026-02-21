"""
Step 5: Optical Performance Analysis Page.
Computes and displays PSF, MTF, Strehl ratio, encircled energy,
Zernike decomposition, and far-field patterns.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QTabWidget, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog
)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from core.design import DesignState
from analysis.psf import compute_psf
from analysis.mtf import compute_mtf
from analysis.strehl import compute_strehl_ratio
from analysis.encircled_energy import compute_encircled_energy
from analysis.zernike import zernike_decomposition, ZERNIKE_NAMES
from analysis.farfield import compute_farfield


class AnalysisPage(QWidget):
    """Optical performance analysis dashboard."""

    def __init__(self, design: DesignState, parent=None):
        super().__init__(parent)
        self.design = design
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Top controls
        top_row = QHBoxLayout()

        top_row.addWidget(QLabel("Focal length (um):"))
        self.focal_spin = QDoubleSpinBox()
        self.focal_spin.setRange(1, 1e6)
        self.focal_spin.setValue(100)
        self.focal_spin.setDecimals(1)
        top_row.addWidget(self.focal_spin)

        top_row.addWidget(QLabel("Propagation dist (um):"))
        self.prop_spin = QDoubleSpinBox()
        self.prop_spin.setRange(1, 1e7)
        self.prop_spin.setValue(100)
        self.prop_spin.setDecimals(1)
        top_row.addWidget(self.prop_spin)

        run_btn = QPushButton("Run Full Analysis")
        run_btn.setObjectName("primary")
        run_btn.clicked.connect(self.run_analysis)
        top_row.addWidget(run_btn)

        export_btn = QPushButton("Export Plots")
        export_btn.setObjectName("accent")
        export_btn.clicked.connect(self._export_plots)
        top_row.addWidget(export_btn)

        top_row.addStretch()
        layout.addLayout(top_row)

        # Metric summary cards
        metrics_row = QHBoxLayout()
        self.strehl_label = self._make_metric_card("Strehl Ratio", "—")
        self.ee84_label = self._make_metric_card("EE @ 84%", "—")
        self.peak_label = self._make_metric_card("PSF Peak", "—")
        metrics_row.addWidget(self.strehl_label[0])
        metrics_row.addWidget(self.ee84_label[0])
        metrics_row.addWidget(self.peak_label[0])
        layout.addLayout(metrics_row)

        # Tab widget for analysis plots
        self.tabs = QTabWidget()

        # PSF tab
        self.psf_fig = Figure(figsize=(8, 3.5), facecolor='#1e2040')
        self.psf_canvas = FigureCanvas(self.psf_fig)
        self.tabs.addTab(self.psf_canvas, "PSF")

        # MTF tab
        self.mtf_fig = Figure(figsize=(8, 3.5), facecolor='#1e2040')
        self.mtf_canvas = FigureCanvas(self.mtf_fig)
        self.tabs.addTab(self.mtf_canvas, "MTF")

        # Encircled Energy tab
        self.ee_fig = Figure(figsize=(8, 3.5), facecolor='#1e2040')
        self.ee_canvas = FigureCanvas(self.ee_fig)
        self.tabs.addTab(self.ee_canvas, "Encircled Energy")

        # Zernike tab
        zernike_widget = QWidget()
        zl = QHBoxLayout(zernike_widget)
        self.zernike_fig = Figure(figsize=(5, 3.5), facecolor='#1e2040')
        self.zernike_canvas = FigureCanvas(self.zernike_fig)
        zl.addWidget(self.zernike_canvas, stretch=3)

        self.zernike_table = QTableWidget()
        self.zernike_table.setColumnCount(3)
        self.zernike_table.setHorizontalHeaderLabels(["Index", "Aberration", "Coefficient"])
        self.zernike_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        zl.addWidget(self.zernike_table, stretch=2)
        self.tabs.addTab(zernike_widget, "Zernike")

        # Far-field tab
        self.ff_fig = Figure(figsize=(8, 3.5), facecolor='#1e2040')
        self.ff_canvas = FigureCanvas(self.ff_fig)
        self.tabs.addTab(self.ff_canvas, "Far-Field")

        layout.addWidget(self.tabs)

    def _make_metric_card(self, title: str, value: str):
        box = QGroupBox(title)
        bl = QVBoxLayout(box)
        lbl = QLabel(value)
        lbl.setObjectName("metric_value")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(lbl)
        return box, lbl

    def _style_ax(self, ax):
        ax.set_facecolor('#1a1b2e')
        ax.tick_params(colors='#808898', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#303560')
        ax.grid(True, alpha=0.15, color='#404580')

    def run_analysis(self):
        if self.design.phase_mask is None:
            return

        phase = self.design.phase_mask
        wl = self.design.wavelength
        ps = self.design.pixel_size
        focal = self.focal_spin.value()
        prop = self.prop_spin.value()

        pad = 4 if phase.shape[0] < 256 else 2

        # --- PSF ---
        psf, x_psf, y_psf = compute_psf(phase, wl, ps, focal, pad_factor=pad)
        self.psf_fig.clear()
        ax1 = self.psf_fig.add_subplot(121)
        ax2 = self.psf_fig.add_subplot(122)
        self._style_ax(ax1)
        self._style_ax(ax2)

        extent = [x_psf[0], x_psf[-1], y_psf[0], y_psf[-1]]
        im = ax1.imshow(psf, cmap='inferno', extent=extent, interpolation='bilinear')
        ax1.set_title('PSF (2D)', color='#e0e0e8', fontsize=11)
        ax1.set_xlabel('x (um)', color='#b0b8d0')
        ax1.set_ylabel('y (um)', color='#b0b8d0')
        self.psf_fig.colorbar(im, ax=ax1, fraction=0.046)

        center = psf.shape[0] // 2
        ax2.plot(x_psf, psf[center, :], color='#5090e0', linewidth=1.5, label='x-cut')
        ax2.plot(y_psf, psf[:, center], color='#ff8060', linewidth=1.5, alpha=0.7, label='y-cut')
        ax2.set_title('PSF Cross-Section', color='#e0e0e8', fontsize=11)
        ax2.set_xlabel('Position (um)', color='#b0b8d0')
        ax2.set_ylabel('Normalized Intensity', color='#b0b8d0')
        ax2.legend(facecolor='#22253e', edgecolor='#404580', labelcolor='#c0c8e0', fontsize=9)

        self.psf_fig.patch.set_facecolor('#1e2040')
        self.psf_fig.tight_layout()
        self.psf_canvas.draw()

        # --- MTF ---
        freqs, mtf_vals, mtf_diff = compute_mtf(psf, ps, wl, focal)
        self.mtf_fig.clear()
        ax_mtf = self.mtf_fig.add_subplot(111)
        self._style_ax(ax_mtf)

        max_freq = freqs[len(freqs)//2] if len(freqs) > 10 else freqs[-1]
        mask_f = freqs <= max_freq
        ax_mtf.plot(freqs[mask_f], mtf_vals[mask_f], color='#5090e0', linewidth=2, label='Metalens MTF')
        ax_mtf.plot(freqs[mask_f], mtf_diff[mask_f], '--', color='#ff8060', linewidth=1.5, label='Diffraction Limit')
        ax_mtf.set_title('Modulation Transfer Function', color='#e0e0e8', fontsize=12)
        ax_mtf.set_xlabel('Spatial Frequency (lp/mm)', color='#b0b8d0')
        ax_mtf.set_ylabel('MTF', color='#b0b8d0')
        ax_mtf.set_ylim(0, 1.1)
        ax_mtf.legend(facecolor='#22253e', edgecolor='#404580', labelcolor='#c0c8e0')

        self.mtf_fig.patch.set_facecolor('#1e2040')
        self.mtf_fig.tight_layout()
        self.mtf_canvas.draw()

        # --- Strehl Ratio ---
        strehl, _, _ = compute_strehl_ratio(phase, pad_factor=pad)
        self.strehl_label[1].setText(f"{strehl:.4f}")
        color = '#6ecf6e' if strehl > 0.8 else '#ffaa40' if strehl > 0.5 else '#ff6b6b'
        self.strehl_label[1].setStyleSheet(f"color: {color}; font-size: 28px; font-weight: bold;")

        # --- Encircled Energy ---
        radii_ee, ee_vals = compute_encircled_energy(psf, ps)
        self.ee_fig.clear()
        ax_ee = self.ee_fig.add_subplot(111)
        self._style_ax(ax_ee)

        ax_ee.plot(radii_ee, ee_vals, color='#5090e0', linewidth=2)
        ax_ee.axhline(0.84, color='#ff8060', linestyle='--', alpha=0.7, label='84% (Airy)')
        idx84 = np.searchsorted(ee_vals, 0.84)
        if idx84 < len(radii_ee):
            r84 = radii_ee[idx84]
            ax_ee.axvline(r84, color='#6ecf6e', linestyle=':', alpha=0.7)
            self.ee84_label[1].setText(f"{r84:.2f} um")
        else:
            self.ee84_label[1].setText("N/A")

        ax_ee.set_title('Encircled Energy', color='#e0e0e8', fontsize=12)
        ax_ee.set_xlabel('Radius (um)', color='#b0b8d0')
        ax_ee.set_ylabel('Encircled Energy Fraction', color='#b0b8d0')
        ax_ee.set_ylim(0, 1.05)
        ax_ee.legend(facecolor='#22253e', edgecolor='#404580', labelcolor='#c0c8e0')

        self.ee_fig.patch.set_facecolor('#1e2040')
        self.ee_fig.tight_layout()
        self.ee_canvas.draw()

        # PSF peak (in context)
        self.peak_label[1].setText(f"{psf.max():.4f}")

        # --- Zernike ---
        coeffs, names = zernike_decomposition(phase, n_terms=15)
        self.zernike_fig.clear()
        ax_z = self.zernike_fig.add_subplot(111)
        self._style_ax(ax_z)

        indices = list(coeffs.keys())
        values = [coeffs[k] for k in indices]
        colors_z = ['#ff6b6b' if v < 0 else '#5090e0' for v in values]
        ax_z.barh(range(len(indices)), values, color=colors_z, edgecolor='#303560')
        ax_z.set_yticks(range(len(indices)))
        ax_z.set_yticklabels([f"Z{j}" for j in indices], fontsize=9)
        ax_z.set_xlabel('Coefficient (rad)', color='#b0b8d0')
        ax_z.set_title('Zernike Decomposition', color='#e0e0e8', fontsize=12)
        ax_z.invert_yaxis()

        self.zernike_fig.patch.set_facecolor('#1e2040')
        self.zernike_fig.tight_layout()
        self.zernike_canvas.draw()

        # Zernike table
        self.zernike_table.setRowCount(len(coeffs))
        for row, (j, val) in enumerate(coeffs.items()):
            self.zernike_table.setItem(row, 0, QTableWidgetItem(f"Z{j}"))
            self.zernike_table.setItem(row, 1, QTableWidgetItem(names[j]))
            item = QTableWidgetItem(f"{val:.4f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.zernike_table.setItem(row, 2, item)

        # --- Far-field ---
        ff_int, ff_x, ff_y = compute_farfield(phase, wl, ps, prop, pad_factor=pad)
        self.ff_fig.clear()
        ax_ff1 = self.ff_fig.add_subplot(121)
        ax_ff2 = self.ff_fig.add_subplot(122)
        self._style_ax(ax_ff1)
        self._style_ax(ax_ff2)

        extent_ff = [ff_x[0], ff_x[-1], ff_y[0], ff_y[-1]]
        im_ff = ax_ff1.imshow(ff_int, cmap='hot', extent=extent_ff, interpolation='bilinear')
        ax_ff1.set_title(f'Far-Field @ z={prop} um', color='#e0e0e8', fontsize=11)
        ax_ff1.set_xlabel('x (um)', color='#b0b8d0')
        ax_ff1.set_ylabel('y (um)', color='#b0b8d0')
        self.ff_fig.colorbar(im_ff, ax=ax_ff1, fraction=0.046)

        ff_log = np.log10(ff_int + 1e-10)
        im_ff2 = ax_ff2.imshow(ff_log, cmap='magma', extent=extent_ff, interpolation='bilinear')
        ax_ff2.set_title('Far-Field (log scale)', color='#e0e0e8', fontsize=11)
        ax_ff2.set_xlabel('x (um)', color='#b0b8d0')
        self.ff_fig.colorbar(im_ff2, ax=ax_ff2, fraction=0.046)

        self.ff_fig.patch.set_facecolor('#1e2040')
        self.ff_fig.tight_layout()
        self.ff_canvas.draw()

    def _export_plots(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return
        try:
            self.psf_fig.savefig(f"{folder}/psf_analysis.png", dpi=200, facecolor='#1e2040')
            self.mtf_fig.savefig(f"{folder}/mtf_analysis.png", dpi=200, facecolor='#1e2040')
            self.ee_fig.savefig(f"{folder}/encircled_energy.png", dpi=200, facecolor='#1e2040')
            self.zernike_fig.savefig(f"{folder}/zernike_decomposition.png", dpi=200, facecolor='#1e2040')
            self.ff_fig.savefig(f"{folder}/farfield_pattern.png", dpi=200, facecolor='#1e2040')
        except Exception as e:
            pass  # silently handle if figure not ready