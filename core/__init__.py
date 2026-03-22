"""
Core Module — Computation Engines
==================================

Contains all computation engines for MetaOpticsAI:
- RCWA electromagnetic simulation
- Phase mask generation
- GDSII layout generation
- FDTD data management
- ML optimization backend
"""

from core.meta_data import MetaDataStore
from core.phase_engine import PhaseEngine
from core.gds_engine import GDSEngine
from core.design import DesignState
from core.rcwa_engine import RCWAEngine

__all__ = [
    'MetaDataStore',
    'PhaseEngine',
    'GDSEngine',
    'DesignState',
    'RCWAEngine',
]
