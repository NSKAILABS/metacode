"""
Target Response Dock
====================

QDockWidget for defining the target optical response that the
optimizer will try to achieve.

Features:
    - Transmittance tab: Define target T(λ) with presets or custom
    - Phase tab: Define target phase profile
    - Multi-objective tab: Weight sliders for different objectives

The target spectrum is visualized with pyqtgraph and can be
imported from CSV files.
"""

import numpy as np
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox, QScrollArea,
    QPushButton, QTabWidget, QSlider, QFileDialog, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    print("Warning: pyqtgraph not available. Install with: pip install pyqtgraph")

from gui.theme import (
    C_BG_MID, C_TEXT_PRIMARY, C_ACCENT_PRIMARY, C_ACCENT_SECONDARY,
    C_SUCCESS, C_WARNING, C_TEXT_MUTED, PLOT_COLORS
)


class TargetResponseDock(QWidget):
    """
    Dock widget for defining the target optical response.
    
    The target response consists of:
        1. Target transmittance spectrum T(λ)
        2. Target phase profile φ(λ)
        3. Multi-objective weights for combined optimization
    
    Provides presets (flat, notch, band-pass, etc.) and allows
    custom definition via CSV import or manual editing.
    """
    
    target_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Wavelength grid
        self._wavelengths = np.linspace(400, 700, 31)
        self._target_T = np.ones(31) * 0.8  # Default: 80% flat transmittance
        self._target_phase = np.zeros(31)   # Default: zero phase
        self._tolerance = 0.05              # Default tolerance band
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Create the dock widget layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Tab widget for different target types
        self.tab_widget = QTabWidget()
        
        # ── Tab 1: Transmittance ──
        trans_tab = QWidget()
        trans_layout = QVBoxLayout(trans_tab)
        trans_layout.setSpacing(12)
        
        # Preset selector
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        
        self.trans_preset_combo = QComboBox()
        self.trans_preset_combo.addItems([
            "Flat (80%)",
            "Flat (95%)",
            "Notch Filter",
            "Band-pass Filter",
            "Gaussian Peak",
            "Linear Gradient",
            "Custom (CSV)",
        ])
        preset_layout.addWidget(self.trans_preset_combo, stretch=1)
        
        self.load_csv_btn = QPushButton("Load CSV...")
        self.load_csv_btn.setVisible(False)
        preset_layout.addWidget(self.load_csv_btn)
        
        trans_layout.addLayout(preset_layout)
        
        # Preset parameters
        self.trans_params_widget = QWidget()
        trans_params_layout = QGridLayout(self.trans_params_widget)
        trans_params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Common parameters (shown/hidden based on preset)
        trans_params_layout.addWidget(QLabel("Center λ:"), 0, 0)
        self.center_wl_spin = QDoubleSpinBox()
        self.center_wl_spin.setRange(200, 2000)
        self.center_wl_spin.setValue(550)
        self.center_wl_spin.setSuffix(" nm")
        trans_params_layout.addWidget(self.center_wl_spin, 0, 1)
        
        trans_params_layout.addWidget(QLabel("Width:"), 1, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(10, 500)
        self.width_spin.setValue(50)
        self.width_spin.setSuffix(" nm")
        trans_params_layout.addWidget(self.width_spin, 1, 1)
        
        trans_params_layout.addWidget(QLabel("Target T:"), 2, 0)
        self.target_t_spin = QDoubleSpinBox()
        self.target_t_spin.setRange(0, 1)
        self.target_t_spin.setValue(0.9)
        self.target_t_spin.setDecimals(2)
        trans_params_layout.addWidget(self.target_t_spin, 2, 1)
        
        trans_layout.addWidget(self.trans_params_widget)
        
        # Tolerance setting
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Tolerance:"))
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.001, 0.5)
        self.tolerance_spin.setValue(0.05)
        self.tolerance_spin.setDecimals(3)
        self.tolerance_spin.setSingleStep(0.01)
        tol_layout.addWidget(self.tolerance_spin)
        tol_layout.addWidget(QLabel("(±)"))
        tol_layout.addStretch()
        trans_layout.addLayout(tol_layout)
        
        # Preview plot
        if HAS_PYQTGRAPH:
            self.trans_plot = pg.PlotWidget()
            self.trans_plot.setBackground(C_BG_MID)
            self.trans_plot.setMinimumHeight(150)
            self.trans_plot.showGrid(x=True, y=True, alpha=0.3)
            self.trans_plot.setLabel('bottom', 'Wavelength', units='nm')
            self.trans_plot.setLabel('left', 'Transmittance')
            self.trans_plot.setYRange(0, 1)
            
            # Target curve
            self.trans_curve = self.trans_plot.plot(
                pen=pg.mkPen(color=PLOT_COLORS['target'], width=2)
            )
            
            # Tolerance fill
            self.trans_fill_upper = self.trans_plot.plot(
                pen=pg.mkPen(color=PLOT_COLORS['target'], width=1, style=Qt.PenStyle.DashLine)
            )
            self.trans_fill_lower = self.trans_plot.plot(
                pen=pg.mkPen(color=PLOT_COLORS['target'], width=1, style=Qt.PenStyle.DashLine)
            )
            
            trans_layout.addWidget(self.trans_plot)
        else:
            trans_layout.addWidget(QLabel("(pyqtgraph not available for preview)"))
        
        self.tab_widget.addTab(trans_tab, "Transmittance")
        
        # ── Tab 2: Phase ──
        phase_tab = QWidget()
        phase_layout = QVBoxLayout(phase_tab)
        phase_layout.setSpacing(12)
        
        # Phase preset selector
        phase_preset_layout = QHBoxLayout()
        phase_preset_layout.addWidget(QLabel("Preset:"))
        
        self.phase_preset_combo = QComboBox()
        self.phase_preset_combo.addItems([
            "Zero (No phase constraint)",
            "Full 2π Coverage",
            "Linear Gradient",
            "Parabolic (Lens)",
            "Custom",
        ])
        phase_preset_layout.addWidget(self.phase_preset_combo, stretch=1)
        phase_layout.addLayout(phase_preset_layout)
        
        # Phase preview plot
        if HAS_PYQTGRAPH:
            self.phase_plot = pg.PlotWidget()
            self.phase_plot.setBackground(C_BG_MID)
            self.phase_plot.setMinimumHeight(150)
            self.phase_plot.showGrid(x=True, y=True, alpha=0.3)
            self.phase_plot.setLabel('bottom', 'Wavelength', units='nm')
            self.phase_plot.setLabel('left', 'Phase', units='rad')
            
            self.phase_curve = self.phase_plot.plot(
                pen=pg.mkPen(color=PLOT_COLORS['secondary'], width=2)
            )
            
            phase_layout.addWidget(self.phase_plot)
        else:
            phase_layout.addWidget(QLabel("(pyqtgraph not available for preview)"))
        
        phase_layout.addStretch()
        self.tab_widget.addTab(phase_tab, "Phase")
        
        # ── Tab 3: Multi-Objective ──
        multi_tab = QWidget()
        multi_layout = QVBoxLayout(multi_tab)
        multi_layout.setSpacing(16)
        
        multi_layout.addWidget(QLabel("Objective Weights:"))
        
        # Transmittance weight
        t_layout = QHBoxLayout()
        t_layout.addWidget(QLabel("Transmittance:"))
        self.t_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.t_weight_slider.setRange(0, 100)
        self.t_weight_slider.setValue(50)
        t_layout.addWidget(self.t_weight_slider, stretch=1)
        self.t_weight_label = QLabel("50%")
        self.t_weight_label.setMinimumWidth(40)
        t_layout.addWidget(self.t_weight_label)
        multi_layout.addLayout(t_layout)
        
        # Phase weight
        p_layout = QHBoxLayout()
        p_layout.addWidget(QLabel("Phase:"))
        self.p_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.p_weight_slider.setRange(0, 100)
        self.p_weight_slider.setValue(30)
        p_layout.addWidget(self.p_weight_slider, stretch=1)
        self.p_weight_label = QLabel("30%")
        self.p_weight_label.setMinimumWidth(40)
        p_layout.addWidget(self.p_weight_label)
        multi_layout.addLayout(p_layout)
        
        # Efficiency weight
        e_layout = QHBoxLayout()
        e_layout.addWidget(QLabel("Efficiency:"))
        self.e_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.e_weight_slider.setRange(0, 100)
        self.e_weight_slider.setValue(20)
        e_layout.addWidget(self.e_weight_slider, stretch=1)
        self.e_weight_label = QLabel("20%")
        self.e_weight_label.setMinimumWidth(40)
        e_layout.addWidget(self.e_weight_label)
        multi_layout.addLayout(e_layout)
        
        # Normalize button
        self.normalize_btn = QPushButton("Normalize to 100%")
        multi_layout.addWidget(self.normalize_btn)
        
        multi_layout.addStretch()
        self.tab_widget.addTab(multi_tab, "Multi-Objective")
        
        main_layout.addWidget(self.tab_widget)
    
    def _connect_signals(self):
        """Connect internal signals."""
        self.trans_preset_combo.currentTextChanged.connect(self._on_trans_preset_changed)
        self.phase_preset_combo.currentTextChanged.connect(self._on_phase_preset_changed)
        
        self.center_wl_spin.valueChanged.connect(self._update_target)
        self.width_spin.valueChanged.connect(self._update_target)
        self.target_t_spin.valueChanged.connect(self._update_target)
        self.tolerance_spin.valueChanged.connect(self._update_target)
        
        self.load_csv_btn.clicked.connect(self._load_csv)
        
        # Weight sliders
        self.t_weight_slider.valueChanged.connect(
            lambda v: self.t_weight_label.setText(f"{v}%")
        )
        self.p_weight_slider.valueChanged.connect(
            lambda v: self.p_weight_label.setText(f"{v}%")
        )
        self.e_weight_slider.valueChanged.connect(
            lambda v: self.e_weight_label.setText(f"{v}%")
        )
        
        self.normalize_btn.clicked.connect(self._normalize_weights)
    
    def _on_trans_preset_changed(self, preset: str):
        """Update UI and target based on selected preset."""
        show_csv = preset == "Custom (CSV)"
        self.load_csv_btn.setVisible(show_csv)
        self.trans_params_widget.setVisible(not show_csv)
        
        self._update_target()
    
    def _on_phase_preset_changed(self, preset: str):
        """Update phase target based on preset."""
        self._update_phase_target()
    
    def _update_target(self):
        """Update target spectrum based on current settings."""
        preset = self.trans_preset_combo.currentText()
        
        if preset == "Flat (80%)":
            self._target_T = np.ones_like(self._wavelengths) * 0.8
        elif preset == "Flat (95%)":
            self._target_T = np.ones_like(self._wavelengths) * 0.95
        elif preset == "Notch Filter":
            center = self.center_wl_spin.value()
            width = self.width_spin.value()
            self._target_T = 1.0 - np.exp(-((self._wavelengths - center) / (width / 2)) ** 2)
        elif preset == "Band-pass Filter":
            center = self.center_wl_spin.value()
            width = self.width_spin.value()
            self._target_T = np.exp(-((self._wavelengths - center) / (width / 2)) ** 2)
        elif preset == "Gaussian Peak":
            center = self.center_wl_spin.value()
            width = self.width_spin.value()
            target_t = self.target_t_spin.value()
            self._target_T = target_t * np.exp(-((self._wavelengths - center) / (width / 2)) ** 2)
        elif preset == "Linear Gradient":
            self._target_T = np.linspace(0.3, 0.9, len(self._wavelengths))
        
        self._tolerance = self.tolerance_spin.value()
        
        # Update plot
        if HAS_PYQTGRAPH:
            self.trans_curve.setData(self._wavelengths, self._target_T)
            self.trans_fill_upper.setData(
                self._wavelengths, 
                np.clip(self._target_T + self._tolerance, 0, 1)
            )
            self.trans_fill_lower.setData(
                self._wavelengths,
                np.clip(self._target_T - self._tolerance, 0, 1)
            )
        
        self.target_changed.emit()
    
    def _update_phase_target(self):
        """Update phase target based on preset."""
        preset = self.phase_preset_combo.currentText()
        
        if preset == "Zero (No phase constraint)":
            self._target_phase = np.zeros_like(self._wavelengths)
        elif preset == "Full 2π Coverage":
            self._target_phase = np.linspace(0, 2 * np.pi, len(self._wavelengths))
        elif preset == "Linear Gradient":
            self._target_phase = np.linspace(0, np.pi, len(self._wavelengths))
        elif preset == "Parabolic (Lens)":
            # Normalize wavelengths to [-1, 1]
            x = (self._wavelengths - self._wavelengths.mean()) / (self._wavelengths.max() - self._wavelengths.min())
            self._target_phase = 2 * np.pi * (1 - x ** 2)
        
        # Update plot
        if HAS_PYQTGRAPH:
            self.phase_curve.setData(self._wavelengths, self._target_phase)
        
        self.target_changed.emit()
    
    def _load_csv(self):
        """Load target spectrum from CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Target Spectrum",
            "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                data = np.loadtxt(file_path, delimiter=',', skiprows=1)
                if data.shape[1] >= 2:
                    self._wavelengths = data[:, 0]
                    self._target_T = data[:, 1]
                    
                    if HAS_PYQTGRAPH:
                        self.trans_curve.setData(self._wavelengths, self._target_T)
                    
                    self.target_changed.emit()
            except Exception as e:
                print(f"Error loading CSV: {e}")
    
    def _normalize_weights(self):
        """Normalize objective weights to sum to 100%."""
        total = (self.t_weight_slider.value() + 
                 self.p_weight_slider.value() + 
                 self.e_weight_slider.value())
        
        if total > 0:
            scale = 100.0 / total
            self.t_weight_slider.setValue(int(self.t_weight_slider.value() * scale))
            self.p_weight_slider.setValue(int(self.p_weight_slider.value() * scale))
            self.e_weight_slider.setValue(int(self.e_weight_slider.value() * scale))
    
    def get_target_spectrum(self) -> Dict[str, Any]:
        """
        Get the target spectrum configuration.
        
        Returns:
            Dictionary containing:
            {
                'wavelengths': np.ndarray,
                'target_T': np.ndarray,
                'target_phase': np.ndarray,
                'tolerance': float,
                'weights': {
                    'transmittance': float,
                    'phase': float,
                    'efficiency': float,
                },
            }
        """
        # Normalize weights
        t_w = self.t_weight_slider.value()
        p_w = self.p_weight_slider.value()
        e_w = self.e_weight_slider.value()
        total = t_w + p_w + e_w
        
        if total > 0:
            weights = {
                'transmittance': t_w / total,
                'phase': p_w / total,
                'efficiency': e_w / total,
            }
        else:
            weights = {'transmittance': 1.0, 'phase': 0.0, 'efficiency': 0.0}
        
        return {
            'wavelengths': self._wavelengths.copy(),
            'target_T': self._target_T.copy(),
            'target_phase': self._target_phase.copy(),
            'tolerance': self._tolerance,
            'weights': weights,
        }
    
    def reset_to_defaults(self):
        """Reset to default values."""
        self._wavelengths = np.linspace(400, 700, 31)
        self._target_T = np.ones(31) * 0.8
        self._target_phase = np.zeros(31)
        self._tolerance = 0.05
        
        self.trans_preset_combo.setCurrentIndex(0)
        self.phase_preset_combo.setCurrentIndex(0)
        
        self.tolerance_spin.setValue(0.05)
        self.center_wl_spin.setValue(550)
        self.width_spin.setValue(50)
        self.target_t_spin.setValue(0.9)
        
        self.t_weight_slider.setValue(50)
        self.p_weight_slider.setValue(30)
        self.e_weight_slider.setValue(20)
        
        self._update_target()
        self._update_phase_target()
