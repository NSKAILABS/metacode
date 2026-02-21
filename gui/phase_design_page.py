"""
Step 3: Phase Mask Design Page.
Generate phase masks using library functions (FZL, Axicon, SPP) or import custom phase image.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QRadioButton, QButtonGroup, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from core.phase_engine import PhaseEngine
from core.design import DesignState


class PhaseDesignPage(QWidget):
    design_ready = pyqtSignal()

    def __init__(self, design: DesignState, parent=None):
        super().__init__(parent)
        self.design = design
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(16)

        # Left: Controls
        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setSpacing(10)

        # Quantization settings
        quant_group = QGroupBox("Quantization Settings")
        qg = QGridLayout()
        qg.setSpacing(6)

        qg.addWidget(QLabel("Unit cells/pixel:"), 0, 0)
        self.ucpp_spin = QSpinBox()
        self.ucpp_spin.setRange(1, 6)
        self.ucpp_spin.setValue(1)
        qg.addWidget(self.ucpp_spin, 0, 1)

        qg.addWidget(QLabel("Phase levels:"), 1, 0)
        self.levels_combo = QComboBox()
        self.levels_combo.addItems(["2", "4", "8", "16", "32"])
        self.levels_combo.setCurrentText("8")
        qg.addWidget(self.levels_combo, 1, 1)

        qg.addWidget(QLabel("Phase min (deg):"), 2, 0)
        self.ph_min_spin = QSpinBox()
        self.ph_min_spin.setRange(0, 359)
        self.ph_min_spin.setValue(0)
        qg.addWidget(self.ph_min_spin, 2, 1)

        qg.addWidget(QLabel("Phase max (deg):"), 3, 0)
        self.ph_max_spin = QSpinBox()
        self.ph_max_spin.setRange(1, 360)
        self.ph_max_spin.setValue(360)
        qg.addWidget(self.ph_max_spin, 3, 1)

        self.circular_check = QCheckBox("Circular aperture")
        self.circular_check.setChecked(True)
        qg.addWidget(self.circular_check, 4, 0, 1, 2)

        quant_group.setLayout(qg)
        ctrl_layout.addWidget(quant_group)

        # Import image button
        import_btn = QPushButton("Import Phase Image")
        import_btn.clicked.connect(self._import_image)
        ctrl_layout.addWidget(import_btn)

        # Library functions
        lib_group = QGroupBox("Library Functions")
        lg = QVBoxLayout()

        self.lib_radio_group = QButtonGroup(self)
        self.rb_fzl = QRadioButton("Fresnel Zone Lens")
        self.rb_axicon = QRadioButton("Axicon")
        self.rb_spp = QRadioButton("Spiral Phase Plate")
        self.rb_fzl.setChecked(True)
        for i, rb in enumerate([self.rb_fzl, self.rb_axicon, self.rb_spp], 1):
            self.lib_radio_group.addButton(rb, i)
            lg.addWidget(rb)

        pg = QGridLayout()
        pg.setSpacing(6)

        pg.addWidget(QLabel("Focal length (um):"), 0, 0)
        self.fzl_f = QDoubleSpinBox()
        self.fzl_f.setRange(1, 1e6)
        self.fzl_f.setValue(100)
        self.fzl_f.setDecimals(1)
        pg.addWidget(self.fzl_f, 0, 1)

        pg.addWidget(QLabel("FZL Diameter (um):"), 1, 0)
        self.fzl_dia = QDoubleSpinBox()
        self.fzl_dia.setRange(1, 1e5)
        self.fzl_dia.setValue(50)
        self.fzl_dia.setDecimals(1)
        pg.addWidget(self.fzl_dia, 1, 1)

        pg.addWidget(QLabel("Cone angle (deg):"), 2, 0)
        self.axi_angle = QDoubleSpinBox()
        self.axi_angle.setRange(0.1, 89)
        self.axi_angle.setValue(5.0)
        pg.addWidget(self.axi_angle, 2, 1)

        pg.addWidget(QLabel("Axicon Dia (um):"), 3, 0)
        self.axi_dia = QDoubleSpinBox()
        self.axi_dia.setRange(1, 1e5)
        self.axi_dia.setValue(50)
        pg.addWidget(self.axi_dia, 3, 1)

        pg.addWidget(QLabel("Topological charge:"), 4, 0)
        self.spp_charge = QSpinBox()
        self.spp_charge.setRange(1, 20)
        self.spp_charge.setValue(1)
        pg.addWidget(self.spp_charge, 4, 1)

        pg.addWidget(QLabel("SPP Diameter (um):"), 5, 0)
        self.spp_dia = QDoubleSpinBox()
        self.spp_dia.setRange(1, 1e5)
        self.spp_dia.setValue(50)
        pg.addWidget(self.spp_dia, 5, 1)

        lg.addLayout(pg)

        gen_btn = QPushButton("Generate Phase Mask")
        gen_btn.setObjectName("primary")
        gen_btn.clicked.connect(self._generate_library)
        lg.addWidget(gen_btn)

        self.status_label = QLabel("")
        lg.addWidget(self.status_label)

        lib_group.setLayout(lg)
        ctrl_layout.addWidget(lib_group)
        ctrl_layout.addStretch()

        layout.addWidget(controls, stretch=2)

        # Right: Phase mask preview
        self.figure = Figure(figsize=(5, 5), facecolor='#1e2040')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(380, 380)
        layout.addWidget(self.canvas, stretch=3)

    def _get_pixel_size(self) -> float:
        return self.ucpp_spin.value() * (self.design.period / 1000.0)

    def _generate_library(self):
        if self.design.period <= 0:
            self._show_status("Set wavelength first (period=0)", error=True)
            return

        pixel_size = self._get_pixel_size()
        if pixel_size <= 0:
            self._show_status("Invalid pixel size", error=True)
            return

        circular = self.circular_check.isChecked()
        wl = self.design.wavelength

        try:
            choice = self.lib_radio_group.checkedId()
            if choice == 1:  # FZL
                phase = PhaseEngine.fresnel_zone_lens(
                    wl, self.fzl_f.value(), self.fzl_dia.value(),
                    pixel_size, circular
                )
                name = "FZL"
            elif choice == 2:  # Axicon
                phase = PhaseEngine.axicon(
                    wl, self.axi_angle.value(), self.axi_dia.value(),
                    pixel_size, circular
                )
                name = "Axicon"
            elif choice == 3:  # SPP
                phase = PhaseEngine.spiral_phase_plate(
                    wl, self.spp_charge.value(), self.spp_dia.value(),
                    pixel_size, circular
                )
                name = "SPP"
            else:
                return

            if phase.shape[0] > 1000:
                self._show_status(
                    f"Mask too large ({phase.shape[0]}px). Reduce diameter or increase unit cells/pixel.",
                    error=True
                )
                return

            self._apply_phase(phase, name)

        except Exception as e:
            self._show_status(f"Error: {e}", error=True)

    def _import_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Phase Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif);;All Files (*)"
        )
        if not filepath:
            return

        try:
            phase = PhaseEngine.from_image(filepath)
            if max(phase.shape) > 1000:
                self._show_status("Image too large (>1000px). Resize first.", error=True)
                return
            self._apply_phase(phase, "Custom Image")
        except Exception as e:
            self._show_status(f"Error loading image: {e}", error=True)

    def _apply_phase(self, phase: np.ndarray, name: str):
        self.design.phase_mask = phase
        self.design.unitcells_per_pixel = self.ucpp_spin.value()
        self.design.phase_levels = int(self.levels_combo.currentText())
        self.design.phase_min = float(self.ph_min_spin.value())
        self.design.phase_max = float(self.ph_max_spin.value())
        self.design.is_circular = self.circular_check.isChecked()

        # Quantize
        quantized, bins = PhaseEngine.quantize_phase(
            phase, self.design.phase_levels,
            self.design.phase_min, self.design.phase_max
        )
        self.design.quantized_mask = quantized
        self.design.bins = bins

        # Plot
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1a1b2e')

        im1 = ax1.imshow(phase, cmap='twilight_shifted', interpolation='nearest')
        ax1.set_title('Continuous Phase', color='#e0e0e8', fontsize=11)
        self.figure.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(quantized, cmap='viridis', interpolation='nearest')
        ax2.set_title(f'Quantized ({self.design.phase_levels} levels)', color='#e0e0e8', fontsize=11)
        self.figure.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        for ax in [ax1, ax2]:
            ax.tick_params(colors='#808898', labelsize=8)

        self.figure.patch.set_facecolor('#1e2040')
        self.figure.tight_layout()
        self.canvas.draw()

        self._show_status(
            f"{name} generated: {phase.shape[0]}x{phase.shape[1]} pixels",
            error=False
        )
        self.design_ready.emit()

    def _show_status(self, msg: str, error: bool = False):
        self.status_label.setText(msg)
        self.status_label.setObjectName("status_error" if error else "status_ok")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)