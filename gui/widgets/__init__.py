"""
Widgets Package
===============

Contains reusable widgets for the MetaOpticsAI application.

Modules:
    - optimization_dashboard: Central dashboard with live pyqtgraph plots
                              for monitoring ML optimization progress
"""

from gui.widgets.optimization_dashboard import (
    OptimizationDashboard,
    StructureViewerWidget,
    StatCard
)

__all__ = [
    'OptimizationDashboard',
    'StructureViewerWidget',
    'StatCard',
]
