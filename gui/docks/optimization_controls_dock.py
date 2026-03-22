"""
Optimization Controls Dock
==========================

QDockWidget containing ML optimizer settings and control buttons.

Features:
    - Backend selection (Meent/PyTorch/JAX)
    - Optimizer selection (Adam/AdamW/SGD/L-BFGS)
    - Learning rate and schedule
    - Training parameters (max iterations, convergence, patience)
    - Regularization (fabrication constraints, TV regularization)
    - Start/Pause/Stop buttons with progress tracking
"""

from typing import Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox, QScrollArea,
    QPushButton, QFrame, QCheckBox, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from gui.theme import (
    C_SUCCESS, C_ACCENT_PRIMARY, C_ACCENT_SECONDARY, 
    C_WARNING, C_ERROR, C_TEXT_MUTED
)


class OptimizationControlsDock(QWidget):
    """
    Dock widget containing all ML optimization settings and controls.
    
    Provides configuration for:
        1. Differentiable backend (Meent/PyTorch/JAX)
        2. Optimizer type and hyperparameters
        3. Training settings (iterations, convergence, patience)
        4. Regularization (fabrication constraints)
        5. Loss function type
    
    Also contains Start/Pause/Stop buttons and progress display.
    """
    
    start_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
    
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
        
        # ── Backend Selection ──
        backend_group = QGroupBox("Differentiable Backend")
        backend_layout = QVBoxLayout(backend_group)
        
        self.backend_combo = QComboBox()
        self.backend_combo.addItems([
            "Meent (Recommended)",
            "Custom PyTorch",
            "Custom JAX",
            "S4 (Non-differentiable)"
        ])
        self.backend_combo.setToolTip(
            "Meent: Differentiable RCWA with GPU support\n"
            "Custom: Bring your own differentiable simulator"
        )
        backend_layout.addWidget(self.backend_combo)
        
        main_layout.addWidget(backend_group)
        
        # ── Optimizer Settings ──
        opt_group = QGroupBox("Optimizer")
        opt_layout = QGridLayout(opt_group)
        opt_layout.setSpacing(8)
        
        opt_layout.addWidget(QLabel("Type:"), 0, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "SGD", "L-BFGS", "RMSprop"])
        opt_layout.addWidget(self.optimizer_combo, 0, 1)
        
        opt_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.001)
        opt_layout.addWidget(self.lr_spin, 1, 1)
        
        opt_layout.addWidget(QLabel("LR Schedule:"), 2, 0)
        self.lr_schedule_combo = QComboBox()
        self.lr_schedule_combo.addItems([
            "Constant",
            "Cosine Annealing",
            "Step Decay",
            "Exponential Decay",
            "ReduceOnPlateau"
        ])
        opt_layout.addWidget(self.lr_schedule_combo, 2, 1)
        
        main_layout.addWidget(opt_group)
        
        # ── Training Parameters ──
        train_group = QGroupBox("Training")
        train_layout = QGridLayout(train_group)
        train_layout.setSpacing(8)
        
        train_layout.addWidget(QLabel("Max Iterations:"), 0, 0)
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 10000)
        self.max_iter_spin.setValue(500)
        self.max_iter_spin.setSingleStep(100)
        train_layout.addWidget(self.max_iter_spin, 0, 1)
        
        train_layout.addWidget(QLabel("Conv. Tolerance:"), 1, 0)
        self.conv_tol_spin = QDoubleSpinBox()
        self.conv_tol_spin.setRange(1e-8, 1e-2)
        self.conv_tol_spin.setValue(1e-5)
        self.conv_tol_spin.setDecimals(8)
        train_layout.addWidget(self.conv_tol_spin, 1, 1)
        
        train_layout.addWidget(QLabel("Patience:"), 2, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(5, 500)
        self.patience_spin.setValue(50)
        self.patience_spin.setToolTip("Stop if no improvement for this many iterations")
        train_layout.addWidget(self.patience_spin, 2, 1)
        
        main_layout.addWidget(train_group)
        
        # ── Regularization ──
        reg_group = QGroupBox("Regularization")
        reg_layout = QVBoxLayout(reg_group)
        reg_layout.setSpacing(8)
        
        self.fab_constraint_check = QCheckBox("Fabrication Constraints")
        self.fab_constraint_check.setChecked(True)
        self.fab_constraint_check.setToolTip("Enforce minimum feature sizes and aspect ratios")
        reg_layout.addWidget(self.fab_constraint_check)
        
        # Min feature size (shown when fab constraints enabled)
        fab_params = QWidget()
        fab_layout = QHBoxLayout(fab_params)
        fab_layout.setContentsMargins(20, 0, 0, 0)
        fab_layout.addWidget(QLabel("Min Feature:"))
        self.min_feature_spin = QDoubleSpinBox()
        self.min_feature_spin.setRange(10, 200)
        self.min_feature_spin.setValue(50)
        self.min_feature_spin.setSuffix(" nm")
        fab_layout.addWidget(self.min_feature_spin)
        fab_layout.addStretch()
        reg_layout.addWidget(fab_params)
        
        self.tv_reg_check = QCheckBox("Total Variation Regularization")
        self.tv_reg_check.setToolTip("Encourage smooth topology boundaries")
        reg_layout.addWidget(self.tv_reg_check)
        
        self.binary_proj_check = QCheckBox("Binary Projection")
        self.binary_proj_check.setToolTip("Push topology towards 0/1 values")
        reg_layout.addWidget(self.binary_proj_check)
        
        main_layout.addWidget(reg_group)
        
        # ── Loss Function ──
        loss_group = QGroupBox("Loss Function")
        loss_layout = QVBoxLayout(loss_group)
        
        self.loss_combo = QComboBox()
        self.loss_combo.addItems([
            "MSE (Mean Squared Error)",
            "MAE (Mean Absolute Error)",
            "Huber Loss",
            "Custom (Python)"
        ])
        loss_layout.addWidget(self.loss_combo)
        
        main_layout.addWidget(loss_group)
        
        # ── Control Buttons ──
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(8)
        
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setObjectName("success")
        self.start_btn.setMinimumHeight(36)
        btn_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setObjectName("warning")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setMinimumHeight(36)
        btn_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(36)
        btn_layout.addWidget(self.stop_btn)
        
        controls_layout.addLayout(btn_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        controls_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(f"color: {C_TEXT_MUTED};")
        controls_layout.addWidget(self.status_label)
        
        main_layout.addWidget(controls_group)
        
        main_layout.addStretch()
        
        scroll.setWidget(container)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
    
    def _connect_signals(self):
        """Connect internal signals."""
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.pause_btn.clicked.connect(self.pause_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        
        self.fab_constraint_check.toggled.connect(
            lambda c: self.min_feature_spin.setEnabled(c)
        )
    
    def set_running(self, running: bool):
        """Update UI state based on whether optimization is running."""
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        
        # Disable settings while running
        for widget in [self.backend_combo, self.optimizer_combo, self.lr_spin,
                       self.lr_schedule_combo, self.max_iter_spin, self.conv_tol_spin,
                       self.patience_spin, self.fab_constraint_check, self.tv_reg_check,
                       self.binary_proj_check, self.loss_combo, self.min_feature_spin]:
            widget.setEnabled(not running)
    
    def set_progress(self, iteration: int, max_iterations: int):
        """Update progress bar."""
        progress = int(100 * iteration / max_iterations) if max_iterations > 0 else 0
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{iteration}/{max_iterations} ({progress}%)")
    
    def set_status(self, message: str, status_type: str = "info"):
        """
        Update status label.
        
        Args:
            message: Status message to display
            status_type: One of 'info', 'success', 'warning', 'error'
        """
        colors = {
            'info': C_TEXT_MUTED,
            'success': C_SUCCESS,
            'warning': C_WARNING,
            'error': C_ERROR,
        }
        color = colors.get(status_type, C_TEXT_MUTED)
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color};")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all optimization parameters as a dictionary.
        
        Returns:
            Dictionary containing:
            {
                'backend': str,
                'optimizer': str,
                'learning_rate': float,
                'lr_schedule': str,
                'max_iterations': int,
                'convergence_tolerance': float,
                'patience': int,
                'fabrication_constraints': bool,
                'min_feature_size': float,
                'tv_regularization': bool,
                'binary_projection': bool,
                'loss_function': str,
            }
        """
        return {
            'backend': self.backend_combo.currentText(),
            'optimizer': self.optimizer_combo.currentText(),
            'learning_rate': self.lr_spin.value(),
            'lr_schedule': self.lr_schedule_combo.currentText(),
            'max_iterations': self.max_iter_spin.value(),
            'convergence_tolerance': self.conv_tol_spin.value(),
            'patience': self.patience_spin.value(),
            'fabrication_constraints': self.fab_constraint_check.isChecked(),
            'min_feature_size': self.min_feature_spin.value(),
            'tv_regularization': self.tv_reg_check.isChecked(),
            'binary_projection': self.binary_proj_check.isChecked(),
            'loss_function': self.loss_combo.currentText(),
        }
