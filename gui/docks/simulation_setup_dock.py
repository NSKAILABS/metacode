"""
Simulation Setup Dock
=====================

QDockWidget containing all RCWA simulation parameters:
    - Illumination settings (wavelength, multi-wavelength sweep)
    - Unit cell geometry (period Tx, Ty)
    - RCWA truncation orders (M, N)
    - Incidence angles (theta, phi)
    - Media refractive indices (input/output)
    - Polarization settings (TE/TM/Unpolarized/Circular/Custom Jones)

Signals:
    parameters_changed: Emitted when any parameter is modified
    wavelength_changed(float): Emitted when wavelength value changes
"""

import numpy as np
from typing import Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox, QScrollArea,
    QCheckBox, QRadioButton, QButtonGroup, QFrame, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal

from gui.theme import (
    C_BG_MID, C_ACCENT_SECONDARY, C_TEXT_MUTED
)


class SimulationSetupDock(QWidget):
    """
    Dock widget containing all simulation/RCWA setup parameters.
    
    This widget provides controls for configuring:
        1. Illumination (wavelength, sweep mode)
        2. Unit cell period (Tx, Ty with optional lock)
        3. RCWA truncation orders (M, N)
        4. Incidence angles (theta, phi)
        5. Media properties (n_input, n_output)
        6. Polarization (TE/TM/Unpolarized/Circular/Custom)
    
    All parameters are accessible via get_parameters() which returns
    a dictionary suitable for passing to the RCWA engine.
    """
    
    # Signals
    parameters_changed = pyqtSignal()
    wavelength_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self.reset_to_defaults()
    
    def _setup_ui(self):
        """Create the dock widget layout."""
        # Main scroll area for all parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Container widget
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(16)
        
        # ── 1. Illumination Group ──
        illum_group = QGroupBox("Illumination")
        illum_layout = QGridLayout(illum_group)
        illum_layout.setSpacing(8)
        
        # Wavelength
        illum_layout.addWidget(QLabel("Wavelength:"), 0, 0)
        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(200, 20000)
        self.wavelength_spin.setValue(532.0)
        self.wavelength_spin.setSuffix(" nm")
        self.wavelength_spin.setDecimals(1)
        illum_layout.addWidget(self.wavelength_spin, 0, 1)
        
        # Multi-wavelength sweep toggle
        self.multi_wl_check = QCheckBox("Multi-λ sweep")
        self.multi_wl_check.setToolTip("Enable wavelength sweep for broadband analysis")
        illum_layout.addWidget(self.multi_wl_check, 1, 0, 1, 2)
        
        # Wavelength range (shown when multi-wl is enabled)
        self.wl_range_widget = QWidget()
        wl_range_layout = QGridLayout(self.wl_range_widget)
        wl_range_layout.setContentsMargins(0, 0, 0, 0)
        
        wl_range_layout.addWidget(QLabel("Start:"), 0, 0)
        self.wl_start_spin = QDoubleSpinBox()
        self.wl_start_spin.setRange(200, 20000)
        self.wl_start_spin.setValue(400.0)
        self.wl_start_spin.setSuffix(" nm")
        wl_range_layout.addWidget(self.wl_start_spin, 0, 1)
        
        wl_range_layout.addWidget(QLabel("End:"), 1, 0)
        self.wl_end_spin = QDoubleSpinBox()
        self.wl_end_spin.setRange(200, 20000)
        self.wl_end_spin.setValue(700.0)
        self.wl_end_spin.setSuffix(" nm")
        wl_range_layout.addWidget(self.wl_end_spin, 1, 1)
        
        wl_range_layout.addWidget(QLabel("Steps:"), 2, 0)
        self.wl_steps_spin = QSpinBox()
        self.wl_steps_spin.setRange(2, 200)
        self.wl_steps_spin.setValue(31)
        wl_range_layout.addWidget(self.wl_steps_spin, 2, 1)
        
        self.wl_range_widget.setVisible(False)
        illum_layout.addWidget(self.wl_range_widget, 2, 0, 1, 2)
        
        main_layout.addWidget(illum_group)
        
        # ── 2. Unit Cell Period Group ──
        period_group = QGroupBox("Unit Cell Period")
        period_layout = QGridLayout(period_group)
        period_layout.setSpacing(8)
        
        period_layout.addWidget(QLabel("Tx (x-period):"), 0, 0)
        self.tx_spin = QDoubleSpinBox()
        self.tx_spin.setRange(10, 100000)
        self.tx_spin.setValue(350.0)
        self.tx_spin.setSuffix(" nm")
        self.tx_spin.setDecimals(1)
        period_layout.addWidget(self.tx_spin, 0, 1)
        
        period_layout.addWidget(QLabel("Ty (y-period):"), 1, 0)
        self.ty_spin = QDoubleSpinBox()
        self.ty_spin.setRange(10, 100000)
        self.ty_spin.setValue(350.0)
        self.ty_spin.setSuffix(" nm")
        self.ty_spin.setDecimals(1)
        period_layout.addWidget(self.ty_spin, 1, 1)
        
        self.lock_period_check = QCheckBox("Lock Tx = Ty")
        self.lock_period_check.setChecked(True)
        period_layout.addWidget(self.lock_period_check, 2, 0, 1, 2)
        
        main_layout.addWidget(period_group)
        
        # ── 3. RCWA Truncation Orders Group ──
        rcwa_group = QGroupBox("RCWA Truncation Orders")
        rcwa_layout = QGridLayout(rcwa_group)
        rcwa_layout.setSpacing(8)
        
        rcwa_layout.addWidget(QLabel("M (x-orders):"), 0, 0)
        self.m_spin = QSpinBox()
        self.m_spin.setRange(1, 30)
        self.m_spin.setValue(5)
        self.m_spin.setToolTip("Number of Fourier orders in x-direction")
        rcwa_layout.addWidget(self.m_spin, 0, 1)
        
        rcwa_layout.addWidget(QLabel("N (y-orders):"), 1, 0)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 30)
        self.n_spin.setValue(5)
        self.n_spin.setToolTip("Number of Fourier orders in y-direction")
        rcwa_layout.addWidget(self.n_spin, 1, 1)
        
        # Info label
        info_label = QLabel("Total orders: (2M+1)×(2N+1)")
        info_label.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px;")
        rcwa_layout.addWidget(info_label, 2, 0, 1, 2)
        
        main_layout.addWidget(rcwa_group)
        
        # ── 4. Incidence Angles Group ──
        angle_group = QGroupBox("Incidence Angles")
        angle_layout = QGridLayout(angle_group)
        angle_layout.setSpacing(8)
        
        angle_layout.addWidget(QLabel("θ (polar):"), 0, 0)
        self.theta_spin = QDoubleSpinBox()
        self.theta_spin.setRange(0, 89)
        self.theta_spin.setValue(0.0)
        self.theta_spin.setSuffix("°")
        self.theta_spin.setDecimals(2)
        angle_layout.addWidget(self.theta_spin, 0, 1)
        
        angle_layout.addWidget(QLabel("φ (azimuthal):"), 1, 0)
        self.phi_spin = QDoubleSpinBox()
        self.phi_spin.setRange(0, 360)
        self.phi_spin.setValue(0.0)
        self.phi_spin.setSuffix("°")
        self.phi_spin.setDecimals(2)
        angle_layout.addWidget(self.phi_spin, 1, 1)
        
        main_layout.addWidget(angle_group)
        
        # ── 5. Media Properties Group ──
        media_group = QGroupBox("Media Properties")
        media_layout = QGridLayout(media_group)
        media_layout.setSpacing(8)
        
        media_layout.addWidget(QLabel("n_input (substrate):"), 0, 0)
        self.n_input_spin = QDoubleSpinBox()
        self.n_input_spin.setRange(1.0, 5.0)
        self.n_input_spin.setValue(1.46)
        self.n_input_spin.setDecimals(3)
        self.n_input_spin.setToolTip("Refractive index of input medium (e.g., SiO2 = 1.46)")
        media_layout.addWidget(self.n_input_spin, 0, 1)
        
        media_layout.addWidget(QLabel("n_output (superstrate):"), 1, 0)
        self.n_output_spin = QDoubleSpinBox()
        self.n_output_spin.setRange(1.0, 5.0)
        self.n_output_spin.setValue(1.0)
        self.n_output_spin.setDecimals(3)
        self.n_output_spin.setToolTip("Refractive index of output medium (e.g., air = 1.0)")
        media_layout.addWidget(self.n_output_spin, 1, 1)
        
        main_layout.addWidget(media_group)
        
        # ── 6. Polarization Group ──
        pol_group = QGroupBox("Polarization")
        pol_layout = QVBoxLayout(pol_group)
        pol_layout.setSpacing(8)
        
        self.pol_button_group = QButtonGroup(self)
        
        self.te_radio = QRadioButton("TE (s-polarized)")
        self.tm_radio = QRadioButton("TM (p-polarized)")
        self.unpol_radio = QRadioButton("Unpolarized")
        self.circular_radio = QRadioButton("Circular (RCP/LCP)")
        self.custom_radio = QRadioButton("Custom Jones vector")
        
        self.pol_button_group.addButton(self.te_radio, 0)
        self.pol_button_group.addButton(self.tm_radio, 1)
        self.pol_button_group.addButton(self.unpol_radio, 2)
        self.pol_button_group.addButton(self.circular_radio, 3)
        self.pol_button_group.addButton(self.custom_radio, 4)
        
        self.te_radio.setChecked(True)
        
        for radio in [self.te_radio, self.tm_radio, self.unpol_radio, 
                      self.circular_radio, self.custom_radio]:
            pol_layout.addWidget(radio)
        
        # Custom Jones vector input (hidden by default)
        self.jones_widget = QWidget()
        jones_layout = QHBoxLayout(self.jones_widget)
        jones_layout.setContentsMargins(20, 0, 0, 0)
        
        jones_layout.addWidget(QLabel("Ex:"))
        self.jones_ex = QLineEdit("1+0j")
        self.jones_ex.setMaximumWidth(80)
        jones_layout.addWidget(self.jones_ex)
        
        jones_layout.addWidget(QLabel("Ey:"))
        self.jones_ey = QLineEdit("0+0j")
        self.jones_ey.setMaximumWidth(80)
        jones_layout.addWidget(self.jones_ey)
        jones_layout.addStretch()
        
        self.jones_widget.setVisible(False)
        pol_layout.addWidget(self.jones_widget)
        
        main_layout.addWidget(pol_group)
        
        # Add stretch at the end
        main_layout.addStretch()
        
        # Set up scroll area
        scroll.setWidget(container)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
    
    def _connect_signals(self):
        """Connect internal signals."""
        # Multi-wavelength toggle
        self.multi_wl_check.toggled.connect(self.wl_range_widget.setVisible)
        self.multi_wl_check.toggled.connect(lambda: self.wavelength_spin.setVisible(
            not self.multi_wl_check.isChecked()
        ))
        
        # Period locking
        self.lock_period_check.toggled.connect(self._on_lock_period_changed)
        self.tx_spin.valueChanged.connect(self._on_tx_changed)
        
        # Wavelength changes
        self.wavelength_spin.valueChanged.connect(
            lambda v: self.wavelength_changed.emit(v)
        )
        
        # Custom Jones vector visibility
        self.custom_radio.toggled.connect(self.jones_widget.setVisible)
        
        # Emit parameters_changed on any change
        for spin in [self.wavelength_spin, self.tx_spin, self.ty_spin,
                     self.m_spin, self.n_spin, self.theta_spin, self.phi_spin,
                     self.n_input_spin, self.n_output_spin,
                     self.wl_start_spin, self.wl_end_spin, self.wl_steps_spin]:
            spin.valueChanged.connect(lambda: self.parameters_changed.emit())
        
        for check in [self.multi_wl_check, self.lock_period_check]:
            check.toggled.connect(lambda: self.parameters_changed.emit())
        
        self.pol_button_group.buttonClicked.connect(
            lambda: self.parameters_changed.emit()
        )
    
    def _on_lock_period_changed(self, locked: bool):
        """Handle period lock checkbox."""
        if locked:
            self.ty_spin.setValue(self.tx_spin.value())
            self.ty_spin.setEnabled(False)
        else:
            self.ty_spin.setEnabled(True)
    
    def _on_tx_changed(self, value: float):
        """Sync Ty to Tx when locked."""
        if self.lock_period_check.isChecked():
            self.ty_spin.blockSignals(True)
            self.ty_spin.setValue(value)
            self.ty_spin.blockSignals(False)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all simulation parameters as a dictionary.
        
        Returns:
            Dictionary containing all parameters suitable for RCWA engine:
            {
                'wavelength': float or np.ndarray,  # nm
                'wavelength_sweep': bool,
                'Tx': float,  # nm
                'Ty': float,  # nm
                'M': int,
                'N': int,
                'theta': float,  # degrees
                'phi': float,  # degrees
                'n_input': float,
                'n_output': float,
                'polarization': str,  # 'TE', 'TM', 'unpolarized', 'circular', 'custom'
                'jones_vector': tuple or None,  # (Ex, Ey) complex numbers
            }
        """
        # Wavelength handling
        if self.multi_wl_check.isChecked():
            wavelength = np.linspace(
                self.wl_start_spin.value(),
                self.wl_end_spin.value(),
                self.wl_steps_spin.value()
            )
            wavelength_sweep = True
        else:
            wavelength = self.wavelength_spin.value()
            wavelength_sweep = False
        
        # Polarization
        pol_id = self.pol_button_group.checkedId()
        pol_map = {0: 'TE', 1: 'TM', 2: 'unpolarized', 3: 'circular', 4: 'custom'}
        polarization = pol_map.get(pol_id, 'TE')
        
        # Jones vector for custom polarization
        jones_vector = None
        if polarization == 'custom':
            try:
                ex = complex(self.jones_ex.text())
                ey = complex(self.jones_ey.text())
                jones_vector = (ex, ey)
            except ValueError:
                jones_vector = (1+0j, 0+0j)  # Default to TE
        
        return {
            'wavelength': wavelength,
            'wavelength_sweep': wavelength_sweep,
            'Tx': self.tx_spin.value(),
            'Ty': self.ty_spin.value(),
            'M': self.m_spin.value(),
            'N': self.n_spin.value(),
            'theta': self.theta_spin.value(),
            'phi': self.phi_spin.value(),
            'n_input': self.n_input_spin.value(),
            'n_output': self.n_output_spin.value(),
            'polarization': polarization,
            'jones_vector': jones_vector,
        }
    
    def set_parameters(self, params: Dict[str, Any]):
        """
        Set simulation parameters from a dictionary.
        
        Args:
            params: Dictionary of parameter values
        """
        if 'wavelength' in params:
            wl = params['wavelength']
            if isinstance(wl, (list, np.ndarray)):
                self.multi_wl_check.setChecked(True)
                self.wl_start_spin.setValue(wl[0])
                self.wl_end_spin.setValue(wl[-1])
                self.wl_steps_spin.setValue(len(wl))
            else:
                self.multi_wl_check.setChecked(False)
                self.wavelength_spin.setValue(wl)
        
        if 'Tx' in params:
            self.tx_spin.setValue(params['Tx'])
        if 'Ty' in params:
            self.ty_spin.setValue(params['Ty'])
        if 'M' in params:
            self.m_spin.setValue(params['M'])
        if 'N' in params:
            self.n_spin.setValue(params['N'])
        if 'theta' in params:
            self.theta_spin.setValue(params['theta'])
        if 'phi' in params:
            self.phi_spin.setValue(params['phi'])
        if 'n_input' in params:
            self.n_input_spin.setValue(params['n_input'])
        if 'n_output' in params:
            self.n_output_spin.setValue(params['n_output'])
    
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        self.wavelength_spin.setValue(532.0)
        self.multi_wl_check.setChecked(False)
        self.wl_start_spin.setValue(400.0)
        self.wl_end_spin.setValue(700.0)
        self.wl_steps_spin.setValue(31)
        
        self.tx_spin.setValue(350.0)
        self.ty_spin.setValue(350.0)
        self.lock_period_check.setChecked(True)
        
        self.m_spin.setValue(5)
        self.n_spin.setValue(5)
        
        self.theta_spin.setValue(0.0)
        self.phi_spin.setValue(0.0)
        
        self.n_input_spin.setValue(1.46)
        self.n_output_spin.setValue(1.0)
        
        self.te_radio.setChecked(True)
        self.jones_ex.setText("1+0j")
        self.jones_ey.setText("0+0j")
