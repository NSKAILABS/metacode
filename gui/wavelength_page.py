"""
Step 1: Wavelength Selection Page.

Allows user to pick from built-in FDTD wavelengths or upload custom data.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QRadioButton, QButtonGroup, QPushButton, QGroupBox,
    QLineEdit, QComboBox, QFileDialog, QMessageBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt
from core.meta_data import MetaDataStore


class WavelengthPage(QWidget):
    """Wavelength selection interface."""

    wavelength_selected = pyqtSignal(int)  # emits selected wavelength

    def __init__(self, data_store: MetaDataStore, parent=None):
        super().__init__(parent)
        self.data_store = data_store
        self._selected_wl = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Select Design Wavelength")
        title.setObjectName("section_header")
        layout.addWidget(title)

        # ── Built-in wavelengths ──
        builtin_group = QGroupBox("Built-in FDTD Library")
        grid = QGridLayout()
        grid.setSpacing(10)

        self.wl_button_group = QButtonGroup(self)

        categories = {
            "Visible": [405, 532, 633, 715],
            "Near-IR": [850, 1064],
            "Mid-IR / THz": [8800, 410000],
        }

        col = 0
        for cat_name, wavelengths in categories.items():
            header = QLabel(cat_name)
            header.setStyleSheet("font-weight: bold; color: #a0c4ff; font-size: 14px;")
            grid.addWidget(header, 0, col, Qt.AlignmentFlag.AlignCenter)

            for row_idx, wl in enumerate(wavelengths):
                entry = self.data_store.get_entry(wl)
                if entry:
                    label_text = self._format_wavelength(wl) + f"  ({entry.shape}, {entry.material})"
                else:
                    label_text = self._format_wavelength(wl)

                rb = QRadioButton(label_text)
                rb.wl_value = wl
                rb.toggled.connect(lambda checked, w=wl: self._on_builtin_selected(w) if checked else None)
                self.wl_button_group.addButton(rb)
                grid.addWidget(rb, row_idx + 1, col)

            col += 1

        builtin_group.setLayout(grid)
        layout.addWidget(builtin_group)

        # ── User data upload ──
        user_group = QGroupBox("Upload Custom FDTD Data")
        ug_layout = QGridLayout()
        ug_layout.setSpacing(8)

        ug_layout.addWidget(QLabel("Shape:"), 0, 0)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Cylinder", "Cross", "Fin"])
        ug_layout.addWidget(self.shape_combo, 0, 1)

        ug_layout.addWidget(QLabel("Wavelength (nm):"), 0, 2)
        self.user_wl_input = QLineEdit("0")
        self.user_wl_input.setMaximumWidth(100)
        ug_layout.addWidget(self.user_wl_input, 0, 3)

        ug_layout.addWidget(QLabel("Period (nm):"), 1, 0)
        self.user_period = QLineEdit("0")
        self.user_period.setMaximumWidth(100)
        ug_layout.addWidget(self.user_period, 1, 1)

        ug_layout.addWidget(QLabel("Height (nm):"), 1, 2)
        self.user_height = QLineEdit("0")
        self.user_height.setMaximumWidth(100)
        ug_layout.addWidget(self.user_height, 1, 3)

        ug_layout.addWidget(QLabel("Cross W / Fin W (nm):"), 2, 0)
        self.user_width = QLineEdit("0")
        self.user_width.setMaximumWidth(100)
        ug_layout.addWidget(self.user_width, 2, 1)

        ug_layout.addWidget(QLabel("Fin L (nm):"), 2, 2)
        self.user_fin_l = QLineEdit("0")
        self.user_fin_l.setMaximumWidth(100)
        ug_layout.addWidget(self.user_fin_l, 2, 3)

        self.upload_btn = QPushButton("📁  Upload Excel File (.xlsx / .xls)")
        self.upload_btn.setObjectName("accent")
        self.upload_btn.clicked.connect(self._upload_user_data)
        ug_layout.addWidget(self.upload_btn, 3, 0, 1, 4)

        self.upload_status = QLabel("")
        ug_layout.addWidget(self.upload_status, 4, 0, 1, 4)

        user_group.setLayout(ug_layout)
        layout.addWidget(user_group)

        layout.addStretch()

    def _format_wavelength(self, wl: int) -> str:
        if wl >= 10000:
            return f"{wl / 1000:.0f} μm"
        elif wl >= 1000:
            return f"{wl} nm"
        else:
            return f"{wl} nm"

    def _on_builtin_selected(self, wl: int):
        self._selected_wl = wl
        self.wavelength_selected.emit(wl)

    def _upload_user_data(self):
        try:
            wl = int(self.user_wl_input.text())
            period = float(self.user_period.text())
            height = float(self.user_height.text())
            shape = self.shape_combo.currentText()

            if wl <= 0 or period <= 0 or height <= 0:
                raise ValueError("Wavelength, period, and height must be positive.")

            filepath, _ = QFileDialog.getOpenFileName(
                self, "Open FDTD Data File", "",
                "Excel Files (*.xlsx *.xls);;All Files (*)"
            )
            if not filepath:
                return

            cross_w = float(self.user_width.text()) if shape in ('Cross', 'Fin') else 0
            fin_l = float(self.user_fin_l.text()) if shape == 'Fin' else 0

            self.data_store.load_from_excel(
                filepath, wl, shape, 'User',
                height, period, cross_w, cross_w, fin_l
            )
            self._selected_wl = wl
            self.wavelength_selected.emit(wl)

            self.upload_status.setText(f"✓ Data loaded for {wl} nm ({shape})")
            self.upload_status.setObjectName("status_ok")
            self.upload_status.style().unpolish(self.upload_status)
            self.upload_status.style().polish(self.upload_status)

        except Exception as e:
            self.upload_status.setText(f"✗ Error: {str(e)}")
            self.upload_status.setObjectName("status_error")
            self.upload_status.style().unpolish(self.upload_status)
            self.upload_status.style().polish(self.upload_status)

    def get_selected_wavelength(self) -> int:
        return self._selected_wl