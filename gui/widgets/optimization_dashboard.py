"""
Optimization Dashboard — Central Widget
========================================

The main visualization area showing live optimization progress with:
    1. Loss curve plot (updating per iteration)
    2. Current vs Target spectrum comparison
    3. Live structure viewer (2D/3D topology evolution)

This widget uses pyqtgraph for high-performance, real-time plotting
suitable for monitoring ML optimization loops.

Usage:
    dashboard = OptimizationDashboard()
    dashboard.start_optimization(sim_params, geom_params, target_params, opt_params)
    
    # Connect to logging
    dashboard.log_message.connect(some_log_function)
"""

import numpy as np
from typing import Dict, Any, Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QFrame, QSplitter, QGroupBox, QSizePolicy, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QBrush

import pyqtgraph as pg

# Import theme colors - works with FLAT project structure
from gui.theme import (
    C_BG_DARKEST, C_BG_DARK, C_BG_MID, C_BG_SURFACE,
    C_TEXT_PRIMARY, C_TEXT_SECONDARY, C_TEXT_MUTED, C_TEXT_ACCENT,
    C_ACCENT_PRIMARY, C_ACCENT_SECONDARY, C_SUCCESS, C_WARNING, C_ERROR,
    C_BORDER_MID, PLOT_COLORS
)


class StructureViewerWidget(QWidget):
    """
    Placeholder widget for 2D/3D structure visualization.
    
    This displays a representation of the metasurface topology evolving
    during optimization. Currently renders a simple 2D grid representation
    that can be replaced with OpenGL/VTK for 3D visualization.
    
    The topology is represented as a grayscale permittivity map where:
        - White (1.0) = High permittivity material (e.g., TiO2)
        - Black (0.0) = Low permittivity material (e.g., air)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Grid resolution for the structure
        self.grid_size = 32
        
        # Initialize with a random topology (will be updated during optimization)
        self._topology = self._generate_initial_topology()
        
        # Target topology for demonstration
        self._target_topology = self._generate_target_topology()
        
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Style
        self.setStyleSheet(f"background-color: {C_BG_MID}; border-radius: 8px;")
    
    def _generate_initial_topology(self) -> np.ndarray:
        """Generate initial random topology."""
        # Start with some random structure (will evolve towards target)
        np.random.seed(42)  # Reproducible for demo
        return np.random.rand(self.grid_size, self.grid_size) * 0.5 + 0.25
    
    def _generate_target_topology(self) -> np.ndarray:
        """Generate target topology (e.g., a circular pillar)."""
        x = np.linspace(-1, 1, self.grid_size)
        y = np.linspace(-1, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Circular pillar in center
        radius = 0.4
        target = (X**2 + Y**2 < radius**2).astype(float)
        return target
    
    def update_topology(self, progress: float):
        """
        Update topology based on optimization progress.
        
        Args:
            progress: Value from 0.0 to 1.0 indicating optimization progress
        """
        # Interpolate between initial and target topology
        progress = np.clip(progress, 0.0, 1.0)
        
        # Add some noise that decreases as we converge
        noise_scale = 0.15 * (1.0 - progress)
        noise = np.random.randn(self.grid_size, self.grid_size) * noise_scale
        
        # Blend towards target
        self._topology = (1 - progress) * self._generate_initial_topology() + \
                         progress * self._target_topology + noise
        self._topology = np.clip(self._topology, 0, 1)
        
        self.update()  # Trigger repaint
    
    def reset_topology(self):
        """Reset to initial random topology."""
        self._topology = self._generate_initial_topology()
        self.update()
    
    def paintEvent(self, event):
        """Paint the topology as a 2D grid."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions with padding
        padding = 20
        width = self.width() - 2 * padding
        height = self.height() - 2 * padding
        
        # Ensure square cells
        cell_size = min(width, height) // self.grid_size
        
        # Calculate offset to center the grid
        offset_x = (self.width() - cell_size * self.grid_size) // 2
        offset_y = (self.height() - cell_size * self.grid_size) // 2
        
        # Draw grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                value = self._topology[i, j]
                
                # Map value to color (dark blue to bright cyan)
                # Low permittivity -> dark, High permittivity -> bright
                r = int(30 + value * 60)
                g = int(40 + value * 120)
                b = int(60 + value * 195)
                
                color = QColor(r, g, b)
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.PenStyle.NoPen)
                
                x = offset_x + j * cell_size
                y = offset_y + i * cell_size
                
                painter.drawRect(x, y, cell_size, cell_size)
        
        # Draw border around the entire grid
        painter.setPen(QPen(QColor(C_BORDER_MID), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(
            offset_x - 1, offset_y - 1,
            cell_size * self.grid_size + 2,
            cell_size * self.grid_size + 2
        )
        
        # Draw "Unit Cell" label
        painter.setPen(QPen(QColor(C_TEXT_MUTED)))
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.drawText(
            offset_x, offset_y - 5,
            "Unit Cell Topology"
        )


class StatCard(QFrame):
    """
    A statistics card displaying a metric value with label.
    
    Used in the dashboard header to show key optimization metrics
    like current loss, best loss, iteration count, etc.
    """
    
    def __init__(self, title: str, value: str = "—", parent=None):
        super().__init__(parent)
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {C_BG_MID};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Value label
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"""
            color: {C_TEXT_PRIMARY};
            font-size: 20px;
            font-weight: bold;
        """)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)
        
        self.setMinimumWidth(100)
    
    def set_value(self, value: str, color: str = None):
        """Update the displayed value."""
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"""
                color: {color};
                font-size: 20px;
                font-weight: bold;
            """)


class OptimizationDashboard(QWidget):
    """
    Central widget for live optimization visualization.
    
    Features:
        - Loss curve plot (pyqtgraph)
        - Current vs Target spectrum comparison (pyqtgraph)
        - Live structure viewer (2D topology evolution)
        - Statistics cards showing key metrics
    
    Signals:
        log_message(str): Emitted when dashboard wants to log something
        optimization_complete(float, int, bool): Emitted when optimization finishes
            Args: final_loss, iterations, converged
    
    Methods:
        start_optimization(sim_params, geom_params, target_params, opt_params)
        stop_optimization()
        toggle_pause()
        clear_history()
    """
    
    # Signals
    log_message = pyqtSignal(str)
    optimization_complete = pyqtSignal(float, int, bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # State Variables
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        self._is_running = False
        self._is_paused = False
        self._current_iteration = 0
        self._max_iterations = 500
        
        # Loss history
        self._loss_history: List[float] = []
        self._best_loss = float('inf')
        self._initial_loss = None
        
        # Spectrum data
        self._wavelengths = np.linspace(400, 700, 61)  # 400-700nm
        self._target_spectrum: Optional[np.ndarray] = None
        self._current_spectrum: Optional[np.ndarray] = None
        
        # Convergence tracking
        self._convergence_tolerance = 1e-5
        self._patience = 50
        self._no_improvement_count = 0
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Timer for Simulation
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._optimization_step)
        self._timer_interval = 50  # ms between updates (20 FPS)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Setup UI
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        self._setup_ui()
        self._initialize_plots()
    
    def _setup_ui(self):
        """Create the dashboard layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # ── Header with Statistics Cards ──
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        self.iteration_card = StatCard("Iteration", "0 / 500")
        self.loss_card = StatCard("Current Loss", "—")
        self.best_loss_card = StatCard("Best Loss", "—")
        self.improvement_card = StatCard("Improvement", "—")
        self.status_card = StatCard("Status", "Ready")
        
        for card in [self.iteration_card, self.loss_card, self.best_loss_card,
                     self.improvement_card, self.status_card]:
            header_layout.addWidget(card)
        
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # ── Main Content Area (Splitter) ──
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {C_BORDER_MID};
                width: 2px;
            }}
        """)
        
        # Left side: Plots (stacked vertically)
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(12)
        
        # Loss Curve Plot
        loss_group = QGroupBox("Loss Curve")
        loss_group.setStyleSheet(self._group_box_style())
        loss_layout = QVBoxLayout(loss_group)
        loss_layout.setContentsMargins(8, 16, 8, 8)
        
        self.loss_plot = pg.PlotWidget()
        self._configure_plot(self.loss_plot, "Iteration", "Loss")
        loss_layout.addWidget(self.loss_plot)
        
        plots_layout.addWidget(loss_group, stretch=1)
        
        # Spectrum Comparison Plot
        spectrum_group = QGroupBox("Current vs Target Spectrum")
        spectrum_group.setStyleSheet(self._group_box_style())
        spectrum_layout = QVBoxLayout(spectrum_group)
        spectrum_layout.setContentsMargins(8, 16, 8, 8)
        
        self.spectrum_plot = pg.PlotWidget()
        self._configure_plot(self.spectrum_plot, "Wavelength (nm)", "Transmittance")
        spectrum_layout.addWidget(self.spectrum_plot)
        
        plots_layout.addWidget(spectrum_group, stretch=1)
        
        content_splitter.addWidget(plots_widget)
        
        # Right side: Structure Viewer
        structure_group = QGroupBox("Live Structure Viewer")
        structure_group.setStyleSheet(self._group_box_style())
        structure_layout = QVBoxLayout(structure_group)
        structure_layout.setContentsMargins(8, 16, 8, 8)
        
        self.structure_viewer = StructureViewerWidget()
        structure_layout.addWidget(self.structure_viewer)
        
        # Info label below structure viewer
        info_label = QLabel(
            "Topology evolves as optimizer adjusts structure parameters.\n"
            "Bright regions = high permittivity material (e.g., TiO₂)"
        )
        info_label.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px;")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        structure_layout.addWidget(info_label)
        
        content_splitter.addWidget(structure_group)
        
        # Set splitter proportions (70% plots, 30% structure)
        content_splitter.setSizes([700, 300])
        
        main_layout.addWidget(content_splitter, stretch=1)
        
        # ── Progress Bar ──
        progress_layout = QHBoxLayout()
        progress_layout.setSpacing(8)
        
        progress_label = QLabel("Progress:")
        progress_label.setStyleSheet(f"color: {C_TEXT_SECONDARY};")
        progress_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (Iteration %v of %m)")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {C_BG_MID};
                border: 1px solid {C_BORDER_MID};
                border-radius: 4px;
                text-align: center;
                color: {C_TEXT_PRIMARY};
                height: 20px;
            }}
            QProgressBar::chunk {{
                background-color: {C_ACCENT_PRIMARY};
                border-radius: 3px;
            }}
        """)
        progress_layout.addWidget(self.progress_bar, stretch=1)
        
        main_layout.addLayout(progress_layout)
    
    def _group_box_style(self) -> str:
        """Return consistent GroupBox styling."""
        return f"""
            QGroupBox {{
                background-color: {C_BG_DARK};
                border: 1px solid {C_BORDER_MID};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
                color: {C_TEXT_ACCENT};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }}
        """
    
    def _configure_plot(self, plot_widget: pg.PlotWidget, x_label: str, y_label: str):
        """Configure a pyqtgraph PlotWidget with consistent styling."""
        plot_widget.setBackground(C_BG_MID)
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Axis styling
        axis_pen = pg.mkPen(color=C_TEXT_MUTED, width=1)
        
        plot_widget.getAxis('bottom').setPen(axis_pen)
        plot_widget.getAxis('left').setPen(axis_pen)
        plot_widget.getAxis('bottom').setTextPen(C_TEXT_SECONDARY)
        plot_widget.getAxis('left').setTextPen(C_TEXT_SECONDARY)
        
        # Labels
        label_style = {'color': C_TEXT_SECONDARY, 'font-size': '11pt'}
        plot_widget.setLabel('bottom', x_label, **label_style)
        plot_widget.setLabel('left', y_label, **label_style)
        
        # Enable mouse interaction
        plot_widget.setMouseEnabled(x=True, y=True)
        plot_widget.enableAutoRange()
    
    def _initialize_plots(self):
        """Initialize plot curves and data."""
        # Loss curve
        self.loss_curve = self.loss_plot.plot(
            pen=pg.mkPen(color=PLOT_COLORS['loss'], width=2),
            name='Loss'
        )
        
        # Best loss reference line
        self.best_loss_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(color=PLOT_COLORS['best'], width=1, style=Qt.PenStyle.DashLine),
            label='Best', labelOpts={'color': C_SUCCESS, 'position': 0.95}
        )
        self.loss_plot.addItem(self.best_loss_line)
        self.best_loss_line.setVisible(False)
        
        # Spectrum curves
        self.target_curve = self.spectrum_plot.plot(
            pen=pg.mkPen(color=PLOT_COLORS['target'], width=2, style=Qt.PenStyle.DashLine),
            name='Target'
        )
        
        self.current_curve = self.spectrum_plot.plot(
            pen=pg.mkPen(color=PLOT_COLORS['current'], width=2),
            name='Current'
        )
        
        # Add fill between for error visualization
        self.error_fill = pg.FillBetweenItem(
            self.target_curve, self.current_curve,
            brush=pg.mkBrush(color=(59, 130, 246, 40))  # Semi-transparent blue
        )
        self.spectrum_plot.addItem(self.error_fill)
        
        # Add legend to spectrum plot
        legend = self.spectrum_plot.addLegend(offset=(60, 10))
        legend.setLabelTextColor(C_TEXT_SECONDARY)
        
        # Initialize with placeholder data
        self._reset_plot_data()
    
    def _reset_plot_data(self):
        """Reset all plot data to initial state."""
        self._loss_history = []
        self._current_iteration = 0
        self._best_loss = float('inf')
        self._initial_loss = None
        
        # Clear loss curve
        self.loss_curve.setData([], [])
        self.best_loss_line.setVisible(False)
        
        # Generate default target spectrum (flat 80% transmittance)
        self._target_spectrum = np.ones_like(self._wavelengths) * 0.8
        
        # Initial current spectrum (random, will converge to target)
        self._current_spectrum = np.random.rand(len(self._wavelengths)) * 0.5 + 0.2
        
        # Update spectrum plot
        self.target_curve.setData(self._wavelengths, self._target_spectrum)
        self.current_curve.setData(self._wavelengths, self._current_spectrum)
        
        # Reset structure viewer
        self.structure_viewer.reset_topology()
        
        # Reset cards
        self.iteration_card.set_value(f"0 / {self._max_iterations}")
        self.loss_card.set_value("—")
        self.best_loss_card.set_value("—")
        self.improvement_card.set_value("—")
        self.status_card.set_value("Ready", C_TEXT_PRIMARY)
        
        # Reset progress
        self.progress_bar.setMaximum(self._max_iterations)
        self.progress_bar.setValue(0)
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def start_optimization(
        self,
        sim_params: Dict[str, Any],
        geom_params: Dict[str, Any],
        target_params: Dict[str, Any],
        opt_params: Dict[str, Any]
    ):
        """
        Start the optimization process.
        
        In the real implementation, this would:
            1. Initialize the RCWA simulator with sim_params
            2. Set up the differentiable geometry with geom_params
            3. Configure the loss function using target_params
            4. Initialize the optimizer with opt_params
            5. Begin the gradient descent loop
        
        For this demo, we simulate the optimization with a QTimer.
        
        Args:
            sim_params: Simulation parameters from SimulationSetupDock
            geom_params: Geometry parameters from GeometryOptimizationDock
            target_params: Target spectrum from TargetResponseDock
            opt_params: Optimizer settings from OptimizationControlsDock
        """
        if self._is_running:
            return
        
        # Reset state
        self._reset_plot_data()
        
        # Extract relevant parameters
        self._max_iterations = opt_params.get('max_iterations', 500)
        self._convergence_tolerance = opt_params.get('convergence_tolerance', 1e-5)
        self._patience = opt_params.get('patience', 50)
        
        # Update progress bar max
        self.progress_bar.setMaximum(self._max_iterations)
        
        # Set target spectrum from target_params
        if 'target_T' in target_params:
            self._target_spectrum = np.array(target_params['target_T'])
            if len(self._target_spectrum) != len(self._wavelengths):
                # Interpolate to match our wavelength grid
                target_wl = target_params.get('wavelengths', self._wavelengths)
                self._target_spectrum = np.interp(
                    self._wavelengths, target_wl, self._target_spectrum
                )
        
        # Update target curve
        self.target_curve.setData(self._wavelengths, self._target_spectrum)
        
        # Initialize current spectrum (random starting point)
        self._current_spectrum = np.random.rand(len(self._wavelengths)) * 0.4 + 0.3
        
        # Calculate initial loss
        self._initial_loss = self._calculate_loss()
        self._best_loss = self._initial_loss
        
        # Start optimization
        self._is_running = True
        self._is_paused = False
        self._no_improvement_count = 0
        
        self.status_card.set_value("Running", C_SUCCESS)
        self.log_message.emit("Optimization started")
        
        # Start the update timer
        self._update_timer.start(self._timer_interval)
    
    def stop_optimization(self):
        """Stop the optimization process."""
        if not self._is_running:
            return
        
        self._update_timer.stop()
        self._is_running = False
        self._is_paused = False
        
        self.status_card.set_value("Stopped", C_WARNING)
        self.log_message.emit(
            f"Optimization stopped at iteration {self._current_iteration}"
        )
    
    def toggle_pause(self):
        """Toggle pause state of optimization."""
        if not self._is_running:
            return
        
        self._is_paused = not self._is_paused
        
        if self._is_paused:
            self._update_timer.stop()
            self.status_card.set_value("Paused", C_WARNING)
            self.log_message.emit("Optimization paused")
        else:
            self._update_timer.start(self._timer_interval)
            self.status_card.set_value("Running", C_SUCCESS)
            self.log_message.emit("Optimization resumed")
    
    def clear_history(self):
        """Clear all optimization history and reset plots."""
        self.stop_optimization()
        self._reset_plot_data()
        self.log_message.emit("History cleared")
    
    # =========================================================================
    # OPTIMIZATION SIMULATION
    # =========================================================================
    
    def _optimization_step(self):
        """
        Perform one optimization step (simulated).
        
        In a real implementation, this would:
            1. Run forward RCWA simulation
            2. Calculate loss against target
            3. Compute gradients via backpropagation
            4. Update parameters using optimizer
            5. Update visualizations
        """
        if self._is_paused:
            return
        
        self._current_iteration += 1
        
        # ── Simulate spectrum converging to target ──
        # Rate of convergence depends on iteration
        progress = self._current_iteration / self._max_iterations
        
        # Add some realistic noise that decreases over time
        noise_scale = 0.05 * (1 - progress)
        noise = np.random.randn(len(self._wavelengths)) * noise_scale
        
        # Exponential convergence towards target
        convergence_rate = 0.02 + 0.01 * np.random.rand()  # Some randomness
        self._current_spectrum = (
            self._current_spectrum * (1 - convergence_rate) +
            self._target_spectrum * convergence_rate +
            noise
        )
        self._current_spectrum = np.clip(self._current_spectrum, 0, 1)
        
        # ── Calculate loss ──
        current_loss = self._calculate_loss()
        self._loss_history.append(current_loss)
        
        # Track best loss
        if current_loss < self._best_loss:
            self._best_loss = current_loss
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
        
        # ── Update visualizations ──
        self._update_plots()
        self._update_stats(current_loss)
        
        # Update structure viewer
        self.structure_viewer.update_topology(progress)
        
        # ── Check stopping conditions ──
        converged = False
        
        # Convergence check
        if len(self._loss_history) > 10:
            recent_improvement = abs(
                self._loss_history[-10] - self._loss_history[-1]
            )
            if recent_improvement < self._convergence_tolerance:
                converged = True
                self._finish_optimization(converged=True)
                return
        
        # Patience check
        if self._no_improvement_count >= self._patience:
            self._finish_optimization(converged=False, reason="patience exceeded")
            return
        
        # Max iterations check
        if self._current_iteration >= self._max_iterations:
            self._finish_optimization(converged=False, reason="max iterations reached")
            return
        
        # Log every 50 iterations
        if self._current_iteration % 50 == 0:
            self.log_message.emit(
                f"Iteration {self._current_iteration}: "
                f"Loss = {current_loss:.6f}, Best = {self._best_loss:.6f}"
            )
    
    def _calculate_loss(self) -> float:
        """
        Calculate MSE loss between current and target spectrum.
        
        Returns:
            Mean squared error loss value
        """
        diff = self._current_spectrum - self._target_spectrum
        mse = np.mean(diff ** 2)
        return mse
    
    def _update_plots(self):
        """Update all plots with current data."""
        # Loss curve
        iterations = np.arange(1, len(self._loss_history) + 1)
        self.loss_curve.setData(iterations, self._loss_history)
        
        # Best loss line
        if self._best_loss < float('inf'):
            self.best_loss_line.setPos(self._best_loss)
            self.best_loss_line.setVisible(True)
        
        # Spectrum plot
        self.current_curve.setData(self._wavelengths, self._current_spectrum)
    
    def _update_stats(self, current_loss: float):
        """Update statistics cards."""
        # Iteration
        self.iteration_card.set_value(
            f"{self._current_iteration} / {self._max_iterations}"
        )
        
        # Progress bar
        self.progress_bar.setValue(self._current_iteration)
        
        # Current loss
        self.loss_card.set_value(f"{current_loss:.6f}")
        
        # Best loss
        self.best_loss_card.set_value(f"{self._best_loss:.6f}", C_SUCCESS)
        
        # Improvement (relative to initial)
        if self._initial_loss is not None and self._initial_loss > 0:
            improvement = (1 - current_loss / self._initial_loss) * 100
            if improvement > 0:
                self.improvement_card.set_value(f"↓ {improvement:.1f}%", C_SUCCESS)
            else:
                self.improvement_card.set_value(f"↑ {abs(improvement):.1f}%", C_ERROR)
    
    def _finish_optimization(self, converged: bool, reason: str = "converged"):
        """Handle optimization completion."""
        self._update_timer.stop()
        self._is_running = False
        
        if converged:
            self.status_card.set_value("Converged!", C_SUCCESS)
            self.log_message.emit(
                f"✓ Optimization converged at iteration {self._current_iteration}"
            )
            self.log_message.emit(f"  Final loss: {self._best_loss:.6f}")
        else:
            self.status_card.set_value("Complete", C_ACCENT_PRIMARY)
            self.log_message.emit(
                f"Optimization finished: {reason} at iteration {self._current_iteration}"
            )
            self.log_message.emit(f"  Best loss: {self._best_loss:.6f}")
        
        # Emit completion signal
        self.optimization_complete.emit(
            self._best_loss,
            self._current_iteration,
            converged
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def is_running(self) -> bool:
        """Check if optimization is currently running."""
        return self._is_running
    
    def is_paused(self) -> bool:
        """Check if optimization is paused."""
        return self._is_paused
    
    def get_best_loss(self) -> float:
        """Get the best loss achieved so far."""
        return self._best_loss
    
    def get_loss_history(self) -> List[float]:
        """Get the complete loss history."""
        return self._loss_history.copy()
    
    def get_current_spectrum(self) -> np.ndarray:
        """Get the current simulated spectrum."""
        return self._current_spectrum.copy() if self._current_spectrum is not None else None
