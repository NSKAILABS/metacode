"""
Geometry Optimization Dock
==========================

QDockWidget containing structure geometry parameters with
trainable parameter toggles for ML-based inverse design.

Features:
    - Structure type selection (Cylinder, Rectangle, Cross, etc.)
    - Per-parameter "Optimize" checkbox with min/max bounds
    - Material selection with refractive index override
    - Real-time gradient display (during optimization)

Each geometry parameter is wrapped in an OptimizableParameter widget
that provides the checkbox, bounds inputs, and gradient display.
"""

import numpy as np
from typing import Dict, Any, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox, QScrollArea,
    QCheckBox, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

from gui.theme import (
    C_SUCCESS, C_ACCENT_SECONDARY, C_TEXT_MUTED, C_WARNING
)


class OptimizableParameter(QWidget):
    """
    A parameter input with optional optimization toggle.
    
    Layout:
        [Label] [SpinBox] [☑ Optimize]
                          Min: [___] Max: [___]
                          Gradient: 0.00123
    
    When "Optimize" is checked, the bounds inputs become visible
    and the parameter is marked as trainable for the ML optimizer.
    """
    
    value_changed = pyqtSignal(float)
    optimize_toggled = pyqtSignal(bool)
    
    def __init__(
        self,
        name: str,
        default_value: float = 100.0,
        min_value: float = 10.0,
        max_value: float = 1000.0,
        suffix: str = " nm",
        decimals: int = 1,
        parent=None
    ):
        super().__init__(parent)
        
        self.param_name = name
        self._gradient_value: Optional[float] = None
        
        self._setup_ui(name, default_value, min_value, max_value, suffix, decimals)
        self._connect_signals()
    
    def _setup_ui(self, name, default_value, min_value, max_value, suffix, decimals):
        """Create the parameter widget layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 4, 0, 4)
        main_layout.setSpacing(4)
        
        # Top row: Label, SpinBox, Optimize checkbox
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        
        self.label = QLabel(f"{name}:")
        self.label.setMinimumWidth(80)
        top_row.addWidget(self.label)
        
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setRange(min_value, max_value)
        self.spin_box.setValue(default_value)
        self.spin_box.setSuffix(suffix)
        self.spin_box.setDecimals(decimals)
        self.spin_box.setMinimumWidth(100)
        top_row.addWidget(self.spin_box)
        
        self.optimize_check = QCheckBox("Optimize")
        self.optimize_check.setObjectName("optimize_checkbox")
        self.optimize_check.setToolTip("Mark this parameter as trainable for optimization")
        top_row.addWidget(self.optimize_check)
        
        top_row.addStretch()
        main_layout.addLayout(top_row)
        
        # Bounds row (hidden by default)
        self.bounds_widget = QWidget()
        bounds_layout = QHBoxLayout(self.bounds_widget)
        bounds_layout.setContentsMargins(88, 0, 0, 0)  # Indent to align with spinbox
        bounds_layout.setSpacing(8)
        
        bounds_layout.addWidget(QLabel("Min:"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(min_value, max_value)
        self.min_spin.setValue(min_value)
        self.min_spin.setSuffix(suffix)
        self.min_spin.setDecimals(decimals)
        self.min_spin.setMaximumWidth(80)
        bounds_layout.addWidget(self.min_spin)
        
        bounds_layout.addWidget(QLabel("Max:"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(min_value, max_value)
        self.max_spin.setValue(max_value)
        self.max_spin.setSuffix(suffix)
        self.max_spin.setDecimals(decimals)
        self.max_spin.setMaximumWidth(80)
        bounds_layout.addWidget(self.max_spin)
        
        # Gradient display (shown during optimization)
        self.gradient_label = QLabel("")
        self.gradient_label.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px;")
        bounds_layout.addWidget(self.gradient_label)
        
        bounds_layout.addStretch()
        
        self.bounds_widget.setVisible(False)
        main_layout.addWidget(self.bounds_widget)
    
    def _connect_signals(self):
        """Connect internal signals."""
        self.spin_box.valueChanged.connect(self.value_changed.emit)
        self.optimize_check.toggled.connect(self._on_optimize_toggled)
    
    def _on_optimize_toggled(self, checked: bool):
        """Handle optimize checkbox toggle."""
        self.bounds_widget.setVisible(checked)
        self.optimize_toggled.emit(checked)
        
        # Update label styling
        if checked:
            self.label.setStyleSheet(f"color: {C_SUCCESS}; font-weight: bold;")
        else:
            self.label.setStyleSheet("")
    
    def get_value(self) -> float:
        """Get current parameter value."""
        return self.spin_box.value()
    
    def set_value(self, value: float):
        """Set parameter value."""
        self.spin_box.setValue(value)
    
    def is_optimizable(self) -> bool:
        """Check if parameter is marked for optimization."""
        return self.optimize_check.isChecked()
    
    def get_bounds(self) -> tuple:
        """Get optimization bounds (min, max)."""
        return (self.min_spin.value(), self.max_spin.value())
    
    def set_bounds(self, min_val: float, max_val: float):
        """Set optimization bounds."""
        self.min_spin.setValue(min_val)
        self.max_spin.setValue(max_val)
    
    def set_gradient(self, gradient: float):
        """Display current gradient value during optimization."""
        self._gradient_value = gradient
        if abs(gradient) > 1e-6:
            sign = "↑" if gradient > 0 else "↓"
            color = C_SUCCESS if gradient < 0 else C_WARNING  # Negative = decreasing loss = good
            self.gradient_label.setText(f"∇: {gradient:+.2e} {sign}")
            self.gradient_label.setStyleSheet(f"color: {color}; font-size: 10px;")
        else:
            self.gradient_label.setText("∇: ~0")
            self.gradient_label.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px;")
    
    def clear_gradient(self):
        """Clear gradient display."""
        self._gradient_value = None
        self.gradient_label.setText("")
    
    def get_config(self) -> Dict[str, Any]:
        """Get full parameter configuration."""
        return {
            'value': self.get_value(),
            'optimize': self.is_optimizable(),
            'bounds': self.get_bounds() if self.is_optimizable() else None,
            'gradient': self._gradient_value,
        }


class GeometryOptimizationDock(QWidget):
    """
    Dock widget for structure geometry with optimization controls.
    
    Contains:
        1. Structure type selector (Cylinder, Rectangle, Cross, etc.)
        2. Structure-specific parameters with OptimizableParameter widgets
        3. Material selection and custom refractive index
        4. Summary of trainable parameters
    
    The widget dynamically shows/hides parameters based on the
    selected structure type.
    """
    
    parameters_changed = pyqtSignal()
    
    # Available structure types and their parameters
    STRUCTURE_TYPES = {
        'Cylinder': ['radius', 'height'],
        'Rectangle': ['width', 'length', 'height'],
        'Cross': ['arm_width', 'arm_length', 'height'],
        'Ring': ['inner_radius', 'outer_radius', 'height'],
        'Ellipse': ['semi_major', 'semi_minor', 'height'],
        'Custom Topology': [],  # Free-form optimization
    }
    
    # Materials with approximate refractive indices at 532nm
    MATERIALS = {
        'TiO₂ (Titanium Dioxide)': 2.49,
        'Si (Silicon)': 4.15,
        'Si₃N₄ (Silicon Nitride)': 2.02,
        'GaN (Gallium Nitride)': 2.38,
        'a-Si:H (Amorphous Silicon)': 4.30,
        'Custom': None,
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._param_widgets: Dict[str, OptimizableParameter] = {}
        
        self._setup_ui()
        self._connect_signals()
        self.reset_to_defaults()
    
    def _setup_ui(self):
        """Create the dock widget layout."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(16)
        
        # ── Structure Type Group ──
        type_group = QGroupBox("Structure Type")
        type_layout = QVBoxLayout(type_group)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(self.STRUCTURE_TYPES.keys())
        type_layout.addWidget(self.type_combo)
        
        main_layout.addWidget(type_group)
        
        # ── Geometry Parameters Group ──
        self.geom_group = QGroupBox("Geometry Parameters")
        self.geom_layout = QVBoxLayout(self.geom_group)
        self.geom_layout.setSpacing(8)
        
        # Create all possible parameter widgets (we'll show/hide as needed)
        param_configs = {
            'radius': ('Radius', 100.0, 20.0, 500.0, ' nm'),
            'height': ('Height', 200.0, 50.0, 2000.0, ' nm'),
            'width': ('Width', 100.0, 20.0, 500.0, ' nm'),
            'length': ('Length', 100.0, 20.0, 500.0, ' nm'),
            'arm_width': ('Arm Width', 50.0, 20.0, 200.0, ' nm'),
            'arm_length': ('Arm Length', 150.0, 50.0, 400.0, ' nm'),
            'inner_radius': ('Inner Radius', 50.0, 20.0, 200.0, ' nm'),
            'outer_radius': ('Outer Radius', 100.0, 50.0, 300.0, ' nm'),
            'semi_major': ('Semi-major', 120.0, 30.0, 400.0, ' nm'),
            'semi_minor': ('Semi-minor', 80.0, 20.0, 300.0, ' nm'),
        }
        
        for param_id, (name, default, min_v, max_v, suffix) in param_configs.items():
            widget = OptimizableParameter(
                name=name,
                default_value=default,
                min_value=min_v,
                max_value=max_v,
                suffix=suffix,
            )
            widget.value_changed.connect(lambda: self.parameters_changed.emit())
            widget.optimize_toggled.connect(lambda: self.parameters_changed.emit())
            self._param_widgets[param_id] = widget
            self.geom_layout.addWidget(widget)
        
        main_layout.addWidget(self.geom_group)
        
        # ── Material Group ──
        material_group = QGroupBox("Material")
        material_layout = QGridLayout(material_group)
        material_layout.setSpacing(8)
        
        material_layout.addWidget(QLabel("Material:"), 0, 0)
        self.material_combo = QComboBox()
        self.material_combo.addItems(self.MATERIALS.keys())
        material_layout.addWidget(self.material_combo, 0, 1)
        
        # Custom refractive index inputs
        self.custom_n_widget = QWidget()
        custom_n_layout = QGridLayout(self.custom_n_widget)
        custom_n_layout.setContentsMargins(0, 8, 0, 0)
        
        custom_n_layout.addWidget(QLabel("n (real):"), 0, 0)
        self.n_real_spin = QDoubleSpinBox()
        self.n_real_spin.setRange(1.0, 6.0)
        self.n_real_spin.setValue(2.5)
        self.n_real_spin.setDecimals(3)
        custom_n_layout.addWidget(self.n_real_spin, 0, 1)
        
        custom_n_layout.addWidget(QLabel("k (imag):"), 1, 0)
        self.n_imag_spin = QDoubleSpinBox()
        self.n_imag_spin.setRange(0.0, 5.0)
        self.n_imag_spin.setValue(0.0)
        self.n_imag_spin.setDecimals(4)
        custom_n_layout.addWidget(self.n_imag_spin, 1, 1)
        
        self.custom_n_widget.setVisible(False)
        material_layout.addWidget(self.custom_n_widget, 1, 0, 1, 2)
        
        # Display current refractive index
        self.n_display = QLabel("n = 2.490")
        self.n_display.setStyleSheet(f"color: {C_TEXT_MUTED};")
        material_layout.addWidget(self.n_display, 2, 0, 1, 2)
        
        main_layout.addWidget(material_group)
        
        # ── Trainable Parameters Summary ──
        self.summary_group = QGroupBox("Trainable Parameters")
        self.summary_layout = QVBoxLayout(self.summary_group)
        
        self.trainable_label = QLabel("None selected")
        self.trainable_label.setStyleSheet(f"color: {C_TEXT_MUTED};")
        self.trainable_label.setWordWrap(True)
        self.summary_layout.addWidget(self.trainable_label)
        
        main_layout.addWidget(self.summary_group)
        
        main_layout.addStretch()
        
        scroll.setWidget(container)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
    
    def _connect_signals(self):
        """Connect internal signals."""
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.material_combo.currentTextChanged.connect(self._on_material_changed)
        self.n_real_spin.valueChanged.connect(self._update_n_display)
        self.n_imag_spin.valueChanged.connect(self._update_n_display)
        
        # Connect all param widgets to update summary
        for widget in self._param_widgets.values():
            widget.optimize_toggled.connect(self._update_trainable_summary)
    
    def _on_type_changed(self, structure_type: str):
        """Update visible parameters based on structure type."""
        visible_params = self.STRUCTURE_TYPES.get(structure_type, [])
        
        for param_id, widget in self._param_widgets.items():
            widget.setVisible(param_id in visible_params)
        
        self.parameters_changed.emit()
        self._update_trainable_summary()
    
    def _on_material_changed(self, material: str):
        """Update refractive index display."""
        is_custom = material == 'Custom'
        self.custom_n_widget.setVisible(is_custom)
        
        if not is_custom:
            n = self.MATERIALS.get(material, 2.5)
            if n is not None:
                self.n_display.setText(f"n = {n:.3f}")
        else:
            self._update_n_display()
        
        self.parameters_changed.emit()
    
    def _update_n_display(self):
        """Update refractive index display for custom material."""
        n_real = self.n_real_spin.value()
        n_imag = self.n_imag_spin.value()
        
        if n_imag > 0.0001:
            self.n_display.setText(f"n = {n_real:.3f} + {n_imag:.4f}i")
        else:
            self.n_display.setText(f"n = {n_real:.3f}")
    
    def _update_trainable_summary(self):
        """Update the trainable parameters summary."""
        structure_type = self.type_combo.currentText()
        visible_params = self.STRUCTURE_TYPES.get(structure_type, [])
        
        trainable = []
        for param_id in visible_params:
            widget = self._param_widgets.get(param_id)
            if widget and widget.is_optimizable():
                trainable.append(widget.param_name)
        
        if trainable:
            self.trainable_label.setText(", ".join(trainable))
            self.trainable_label.setStyleSheet(f"color: {C_SUCCESS}; font-weight: bold;")
        else:
            self.trainable_label.setText("None selected")
            self.trainable_label.setStyleSheet(f"color: {C_TEXT_MUTED};")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all geometry parameters as a dictionary.
        
        Returns:
            Dictionary containing:
            {
                'structure_type': str,
                'radius': {'value': float, 'optimize': bool, 'bounds': tuple or None},
                ... (other params based on structure type)
                'material': str,
                'n_complex': complex,
                'trainable_params': List[str],  # IDs of optimizable params
            }
        """
        structure_type = self.type_combo.currentText()
        visible_params = self.STRUCTURE_TYPES.get(structure_type, [])
        
        result = {
            'structure_type': structure_type,
            'trainable_params': [],
        }
        
        # Collect parameter values
        for param_id in visible_params:
            widget = self._param_widgets.get(param_id)
            if widget:
                result[param_id] = widget.get_config()
                if widget.is_optimizable():
                    result['trainable_params'].append(param_id)
        
        # Material
        material = self.material_combo.currentText()
        result['material'] = material
        
        if material == 'Custom':
            n_complex = complex(self.n_real_spin.value(), self.n_imag_spin.value())
        else:
            n = self.MATERIALS.get(material, 2.5)
            n_complex = complex(n, 0) if n else complex(2.5, 0)
        
        result['n_complex'] = n_complex
        
        return result
    
    def get_trainable_param_ids(self) -> List[str]:
        """Get list of parameter IDs marked for optimization."""
        structure_type = self.type_combo.currentText()
        visible_params = self.STRUCTURE_TYPES.get(structure_type, [])
        
        trainable = []
        for param_id in visible_params:
            widget = self._param_widgets.get(param_id)
            if widget and widget.is_optimizable():
                trainable.append(param_id)
        
        return trainable
    
    def set_gradient(self, param_id: str, gradient: float):
        """Set gradient display for a parameter during optimization."""
        widget = self._param_widgets.get(param_id)
        if widget:
            widget.set_gradient(gradient)
    
    def clear_gradients(self):
        """Clear all gradient displays."""
        for widget in self._param_widgets.values():
            widget.clear_gradient()
    
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        self.type_combo.setCurrentIndex(0)  # Cylinder
        self.material_combo.setCurrentIndex(0)  # TiO2
        
        self.n_real_spin.setValue(2.5)
        self.n_imag_spin.setValue(0.0)
        
        # Reset all param widgets
        for widget in self._param_widgets.values():
            widget.optimize_check.setChecked(False)
            widget.clear_gradient()
        
        self._on_type_changed('Cylinder')
