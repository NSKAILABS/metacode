"""
Dock Widgets Package
====================

Contains all QDockWidget-based panels for the MetaOpticsAI application.

Modules:
    - simulation_setup_dock: RCWA simulation parameters (wavelength, pitch, etc.)
    - geometry_dock: Structure geometry with trainable parameter toggles
    - target_response_dock: Target spectrum editor with pyqtgraph visualization
    - optimization_controls_dock: ML optimizer settings and control buttons
    - log_dock: Console output and logging display
"""

from gui.docks.simulation_setup_dock import SimulationSetupDock
from gui.docks.geometry_dock import GeometryOptimizationDock, OptimizableParameter
from gui.docks.target_response_dock import TargetResponseDock
from gui.docks.optimization_controls_dock import OptimizationControlsDock
from gui.docks.log_dock import LogDock

__all__ = [
    'SimulationSetupDock',
    'GeometryOptimizationDock',
    'OptimizableParameter',
    'TargetResponseDock',
    'OptimizationControlsDock',
    'LogDock',
]
