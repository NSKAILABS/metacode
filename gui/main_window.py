"""
Main Window — Dockable GUI Architecture
========================================

The primary QMainWindow for MetaOpticsAI v2.0 with a fully dockable interface.

Layout:
    ┌────────────────────────────────────────────────────────────────┐
    │  Menu Bar: File | View | Optimization | Help                  │
    ├──────────────┬─────────────────────────────┬───────────────────┤
    │ Simulation   │   OPTIMIZATION DASHBOARD    │  Target Response  │
    │   Setup      │     (Central Widget)        │                   │
    │   [DOCK]     │   - Loss Curve              │     [DOCK]        │
    │              │   - Spectrum Comparison     │                   │
    ├──────────────┤   - Structure Viewer        ├───────────────────┤
    │  Geometry    │                             │  Optimization     │
    │ Optimization │                             │    Controls       │
    │   [DOCK]     │                             │     [DOCK]        │
    ├──────────────┴─────────────────────────────┴───────────────────┤
    │  Log Console [DOCK]                                            │
    └────────────────────────────────────────────────────────────────┘

Features:
    - All panels are QDockWidgets (draggable, tabbable, floatable)
    - View menu toggles dock visibility
    - Reset Layout restores default arrangement
    - Keyboard shortcuts (F5 = Start, Shift+F5 = Stop)
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QLabel, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QFont, QIcon

# Import dock widgets
from gui.docks.simulation_setup_dock import SimulationSetupDock
from gui.docks.geometry_dock import GeometryOptimizationDock
from gui.docks.target_response_dock import TargetResponseDock
from gui.docks.optimization_controls_dock import OptimizationControlsDock
from gui.docks.log_dock import LogDock

# Import central widget
from gui.widgets.optimization_dashboard import OptimizationDashboard

# Theme colors
from gui.theme import (
    C_ACCENT_PRIMARY, C_SUCCESS, C_WARNING, C_ERROR, C_TEXT_MUTED
)


class MetaOpticsMainWindow(QMainWindow):
    """
    Main application window with dockable panels.
    
    The window consists of:
        - Central widget: OptimizationDashboard (live plots)
        - Left docks: SimulationSetup + GeometryOptimization (tabbed)
        - Right docks: TargetResponse + OptimizationControls (tabbed)
        - Bottom dock: LogConsole
    
    Signals:
        optimization_started: Emitted when optimization begins
        optimization_stopped: Emitted when optimization ends
    """
    
    optimization_started = pyqtSignal()
    optimization_stopped = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("MetaOpticsAI — Inverse Design Platform")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Enable dock nesting and tabbing
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.DockOption.AllowTabbedDocks |
            QMainWindow.DockOption.AllowNestedDocks |
            QMainWindow.DockOption.AnimatedDocks
        )
        
        # Setup UI components
        self._setup_menu_bar()
        self._setup_status_bar()
        self._setup_central_widget()
        self._setup_dock_widgets()
        self._setup_connections()
        
        # Store default state for layout reset
        self._default_state = self.saveState()
    
    def _setup_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # ── File Menu ──
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save Project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_gds_action = QAction("Export &GDS...", self)
        export_gds_action.setShortcut(QKeySequence("Ctrl+E"))
        file_menu.addAction(export_gds_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # ── View Menu ──
        self.view_menu = menubar.addMenu("&View")
        # Dock toggle actions will be added after docks are created
        
        # ── Optimization Menu ──
        opt_menu = menubar.addMenu("&Optimization")
        
        self.start_action = QAction("▶ &Start Optimization", self)
        self.start_action.setShortcut(QKeySequence("F5"))
        self.start_action.triggered.connect(self._start_optimization)
        opt_menu.addAction(self.start_action)
        
        self.stop_action = QAction("■ S&top Optimization", self)
        self.stop_action.setShortcut(QKeySequence("Shift+F5"))
        self.stop_action.triggered.connect(self._stop_optimization)
        self.stop_action.setEnabled(False)
        opt_menu.addAction(self.stop_action)
        
        opt_menu.addSeparator()
        
        clear_history_action = QAction("&Clear History", self)
        clear_history_action.triggered.connect(self._clear_history)
        opt_menu.addAction(clear_history_action)
        
        # ── Help Menu ──
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About MetaOpticsAI", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        docs_action = QAction("&Documentation", self)
        docs_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        help_menu.addAction(docs_action)
    
    def _setup_status_bar(self):
        """Create the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Optimization status indicator
        self.opt_status_label = QLabel("● Ready")
        self.opt_status_label.setStyleSheet(f"color: {C_TEXT_MUTED};")
        self.statusbar.addPermanentWidget(self.opt_status_label)
        
        self.statusbar.showMessage("Welcome to MetaOpticsAI v2.0")
    
    def _setup_central_widget(self):
        """Create the central optimization dashboard."""
        self.dashboard = OptimizationDashboard(parent=self)
        self.setCentralWidget(self.dashboard)
    
    def _setup_dock_widgets(self):
        """Create and position all dock widgets."""
        
        # ── Left Docks ──
        
        # Simulation Setup Dock
        self.sim_setup_dock = QDockWidget("Simulation Setup", self)
        self.sim_setup_dock.setObjectName("sim_setup_dock")
        self.sim_setup_widget = SimulationSetupDock()
        self.sim_setup_dock.setWidget(self.sim_setup_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.sim_setup_dock)
        
        # Geometry Optimization Dock
        self.geom_dock = QDockWidget("Geometry Optimization", self)
        self.geom_dock.setObjectName("geom_dock")
        self.geom_widget = GeometryOptimizationDock()
        self.geom_dock.setWidget(self.geom_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.geom_dock)
        
        # Tab left docks together
        self.tabifyDockWidget(self.sim_setup_dock, self.geom_dock)
        self.sim_setup_dock.raise_()  # Show sim setup on top
        
        # ── Right Docks ──
        
        # Target Response Dock
        self.target_dock = QDockWidget("Target Response", self)
        self.target_dock.setObjectName("target_dock")
        self.target_widget = TargetResponseDock()
        self.target_dock.setWidget(self.target_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.target_dock)
        
        # Optimization Controls Dock
        self.opt_controls_dock = QDockWidget("Optimization Controls", self)
        self.opt_controls_dock.setObjectName("opt_controls_dock")
        self.opt_controls_widget = OptimizationControlsDock()
        self.opt_controls_dock.setWidget(self.opt_controls_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.opt_controls_dock)
        
        # Tab right docks together
        self.tabifyDockWidget(self.target_dock, self.opt_controls_dock)
        self.target_dock.raise_()  # Show target on top
        
        # ── Bottom Dock ──
        
        # Log Console Dock
        self.log_dock = QDockWidget("Log Console", self)
        self.log_dock.setObjectName("log_dock")
        self.log_widget = LogDock()
        self.log_dock.setWidget(self.log_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        
        # Set initial dock sizes
        self.resizeDocks(
            [self.sim_setup_dock, self.target_dock],
            [350, 350],
            Qt.Orientation.Horizontal
        )
        self.resizeDocks(
            [self.log_dock],
            [180],
            Qt.Orientation.Vertical
        )
        
        # ── Add dock toggle actions to View menu ──
        self.view_menu.addAction(self.sim_setup_dock.toggleViewAction())
        self.view_menu.addAction(self.geom_dock.toggleViewAction())
        self.view_menu.addAction(self.target_dock.toggleViewAction())
        self.view_menu.addAction(self.opt_controls_dock.toggleViewAction())
        self.view_menu.addAction(self.log_dock.toggleViewAction())
        
        self.view_menu.addSeparator()
        
        reset_layout_action = QAction("&Reset Layout", self)
        reset_layout_action.triggered.connect(self._reset_layout)
        self.view_menu.addAction(reset_layout_action)
    
    def _setup_connections(self):
        """Connect signals between components."""
        # Dashboard log messages go to log widget
        self.dashboard.log_message.connect(self.log_widget.log_info)
        
        # Optimization controls
        self.opt_controls_widget.start_requested.connect(self._start_optimization)
        self.opt_controls_widget.pause_requested.connect(self._pause_optimization)
        self.opt_controls_widget.stop_requested.connect(self._stop_optimization)
        
        # Dashboard completion
        self.dashboard.optimization_complete.connect(self._on_optimization_complete)
    
    def _start_optimization(self):
        """Start the optimization process."""
        if self.dashboard.is_running():
            return
        
        # Collect parameters from all docks
        sim_params = self.sim_setup_widget.get_parameters()
        geom_params = self.geom_widget.get_parameters()
        target_params = self.target_widget.get_target_spectrum()
        opt_params = self.opt_controls_widget.get_parameters()
        
        # Check for trainable parameters
        if not geom_params.get('trainable_params'):
            QMessageBox.warning(
                self,
                "No Trainable Parameters",
                "Please check at least one 'Optimize' checkbox in the "
                "Geometry panel to mark parameters as trainable."
            )
            return
        
        # Log start
        self.log_widget.log_optimization_start(opt_params)
        
        # Update UI
        self.opt_controls_widget.set_running(True)
        self.start_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self._update_status_indicator("running")
        
        # Start optimization
        self.dashboard.start_optimization(
            sim_params, geom_params, target_params, opt_params
        )
        
        self.optimization_started.emit()
    
    def _pause_optimization(self):
        """Pause/resume the optimization."""
        self.dashboard.toggle_pause()
        
        if self.dashboard.is_paused():
            self._update_status_indicator("paused")
        else:
            self._update_status_indicator("running")
    
    def _stop_optimization(self):
        """Stop the optimization process."""
        self.dashboard.stop_optimization()
        
        # Update UI
        self.opt_controls_widget.set_running(False)
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self._update_status_indicator("stopped")
        
        self.optimization_stopped.emit()
    
    def _on_optimization_complete(self, final_loss: float, iterations: int, converged: bool):
        """Handle optimization completion."""
        # Log completion
        self.log_widget.log_optimization_complete(final_loss, iterations, converged)
        
        # Update UI
        self.opt_controls_widget.set_running(False)
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        
        if converged:
            self._update_status_indicator("converged")
        else:
            self._update_status_indicator("complete")
        
        self.optimization_stopped.emit()
    
    def _clear_history(self):
        """Clear optimization history."""
        self.dashboard.clear_history()
        self.log_widget.clear()
        self.geom_widget.clear_gradients()
        self._update_status_indicator("ready")
    
    def _update_status_indicator(self, status: str):
        """Update the status bar indicator."""
        colors = {
            'ready': C_TEXT_MUTED,
            'running': C_SUCCESS,
            'paused': C_WARNING,
            'stopped': C_WARNING,
            'converged': C_SUCCESS,
            'complete': C_ACCENT_PRIMARY,
            'error': C_ERROR,
        }
        labels = {
            'ready': "● Ready",
            'running': "● Running",
            'paused': "● Paused",
            'stopped': "● Stopped",
            'converged': "● Converged",
            'complete': "● Complete",
            'error': "● Error",
        }
        
        color = colors.get(status, C_TEXT_MUTED)
        label = labels.get(status, "● Ready")
        
        self.opt_status_label.setText(label)
        self.opt_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    def _reset_layout(self):
        """Reset dock layout to default."""
        self.restoreState(self._default_state)
        self.statusbar.showMessage("Layout reset to default", 3000)
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About MetaOpticsAI",
            "<h2>MetaOpticsAI v2.0</h2>"
            "<p><b>AI-Automated Photonics Design Platform</b></p>"
            "<p>Inverse design of metalenses and metasurfaces using "
            "differentiable RCWA simulation and ML optimization.</p>"
            "<hr>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>RCWA electromagnetic simulation (MAXIM formulation)</li>"
            "<li>Gradient-based inverse design with PyTorch/JAX</li>"
            "<li>Real-time optimization visualization</li>"
            "<li>GDSII export for fabrication</li>"
            "</ul>"
            "<hr>"
            "<p>Developed at NIT Hamirpur under NSK AI Labs</p>"
            "<p>© 2024 Dishant & Unisole Empower</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close."""
        if self.dashboard.is_running():
            reply = QMessageBox.question(
                self,
                "Optimization Running",
                "Optimization is still running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            
            self.dashboard.stop_optimization()
        
        event.accept()
