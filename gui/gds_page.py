"""
Step 4: GDSII Generation Page.
Generates fabrication-ready GDSII layout from the quantized phase mask.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGroupBox, QGridLayout, QFileDialog
)
from PyQt6.QtCore import pyqtSignal, Qt, QThread
import numpy as np

from core.gds_engine import GDSEngine
from core.design import DesignState


class GDSWorker(QThread):
    """Background thread for GDS generation."""
    progress = pyqtSignal(float)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, design: DesignState, output_path: str):
        super().__init__()
        self.design = design
        self.output_path = output_path

    def run(self):
        try:
            engine = GDSEngine()
            engine.set_progress_callback(lambda pct: self.progress.emit(pct))
            result = engine.generate(
                quantized_mask=self.design.quantized_mask,
                bins=self.design.bins,
                dimensions=self.design.dimensions,
                phases=self.design.phases,
                period=self.design.period,
                pixel_size_um=self.design.pixel_size,
                unitcells_per_pixel=self.design.unitcells_per_pixel,
                shape=self.design.shape,
                is_circular=self.design.is_circular,
                output_path=self.output_path,
                cross_width=self.design.cross_width,
                fin_width=self.design.fin_width,
                fin_length=self.design.fin_length,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class GDSPage(QWidget):
    """GDSII generation interface with progress tracking."""

    def __init__(self, design: DesignState, parent=None):
        super().__init__(parent)
        self.design = design
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Generate GDSII Layout")
        title.setObjectName("section_header")
        layout.addWidget(title)

        # Design summary
        self.summary_group = QGroupBox("Design Summary")
        sg = QGridLayout()
        sg.setSpacing(6)

        self.info_labels = {}
        info_items = [
            ("wl", "Wavelength"), ("shape", "Shape"), ("period", "Period"),
            ("levels", "Phase Levels"), ("ucpp", "Unit cells/pixel"),
            ("pixel_size", "Pixel Size"), ("mask_size", "Mask Size"),
            ("mask_radius", "Mask Radius"), ("circular", "Circular Aperture"),
        ]
        for idx, (key, name) in enumerate(info_items):
            lbl = QLabel(f"{name}:")
            lbl.setObjectName("param_label")
            val = QLabel("—")
            self.info_labels[key] = val
            sg.addWidget(lbl, idx, 0, Qt.AlignmentFlag.AlignRight)
            sg.addWidget(val, idx, 1)

        self.summary_group.setLayout(sg)
        layout.addWidget(self.summary_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Generate button
        btn_row = QHBoxLayout()
        self.gen_btn = QPushButton("Generate GDS File")
        self.gen_btn.setObjectName("primary")
        self.gen_btn.clicked.connect(self._start_generation)
        btn_row.addWidget(self.gen_btn)
        layout.addLayout(btn_row)

        # Result
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        layout.addStretch()

    def refresh_summary(self):
        d = self.design
        if d.quantized_mask is None:
            return

        self.info_labels["wl"].setText(f"{d.wavelength} nm")
        self.info_labels["shape"].setText(d.shape)
        self.info_labels["period"].setText(f"{d.period} nm")
        self.info_labels["levels"].setText(str(d.phase_levels))
        self.info_labels["ucpp"].setText(f"{d.unitcells_per_pixel}x{d.unitcells_per_pixel}")
        ps = d.pixel_size
        self.info_labels["pixel_size"].setText(f"{d.unitcells_per_pixel * d.period:.0f} nm ({ps:.2f} um)")
        rows, cols = d.quantized_mask.shape
        self.info_labels["mask_size"].setText(f"{rows} x {cols} pixels")
        radius_um = min(rows, cols) * ps / 2
        self.info_labels["mask_radius"].setText(f"{radius_um:.1f} um")
        self.info_labels["circular"].setText("Yes" if d.is_circular else "No")

    def _start_generation(self):
        if self.design.quantized_mask is None:
            self.result_label.setText("No phase mask designed yet.")
            self.result_label.setObjectName("status_error")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save GDS File", "metalens.gds", "GDS Files (*.gds)"
        )
        if not filepath:
            return

        self.gen_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.result_label.setText("Generating...")

        self._worker = GDSWorker(self.design, filepath)
        self._worker.progress.connect(lambda v: self.progress_bar.setValue(int(v)))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, result: dict):
        self.gen_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        elapsed = result['time_seconds']
        mins, secs = int(elapsed // 60), int(elapsed % 60)
        size_kb = result['file_size_kb']
        self.result_label.setText(
            f"GDS generated successfully!\n"
            f"Time: {mins}m {secs}s  |  Size: {size_kb:.1f} KB\n"
            f"Saved to: {result['filepath']}"
        )
        self.result_label.setObjectName("status_ok")
        self.result_label.style().unpolish(self.result_label)
        self.result_label.style().polish(self.result_label)
        self.design.gds_filepath = result['filepath']

    def _on_error(self, msg: str):
        self.gen_btn.setEnabled(True)
        self.result_label.setText(f"Error: {msg}")
        self.result_label.setObjectName("status_error")
        self.result_label.style().unpolish(self.result_label)
        self.result_label.style().polish(self.result_label)