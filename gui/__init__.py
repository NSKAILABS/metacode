"""
GUI Package — PyQt6 Interface
==============================

Contains all GUI components for the MetaOpticsAI application.

New Dockable Architecture (v2.0):
    - main_window: QMainWindow with dock management
    - docks/: Parameter panels (simulation, geometry, target, controls, log)
    - widgets/: Reusable widgets (optimization dashboard)

Legacy Wizard Pages (kept for compatibility):
    - wavelength_page, phase_curve_page, phase_design_page
    - gds_page, analysis_page, rcwa_page
"""

from gui.main_window import MetaOpticsMainWindow
from gui.theme import DARK_STYLESHEET, configure_pyqtgraph_theme, PLOT_COLORS

__all__ = [
    'MetaOpticsMainWindow',
    'DARK_STYLESHEET',
    'configure_pyqtgraph_theme',
    'PLOT_COLORS',
]
