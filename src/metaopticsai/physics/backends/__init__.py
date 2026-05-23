"""Simulation backends — adapter implementations.

The only place in the entire codebase allowed to import `vendor.metabox3`
is `metabox.py`. The import-linter contract in pyproject.toml enforces this.
"""
from metaopticsai.physics.backends.base import SimulationBackend, BackendRegistry
from metaopticsai.physics.backends.analytical import AnalyticalBackend
from metaopticsai.physics.backends.metabox import MetaboxBackend
from metaopticsai.physics.backends.s4 import S4Backend

def default_registry() -> BackendRegistry:
    """Build a registry with all known backends. Availability is auto-detected."""
    reg = BackendRegistry()
    reg.register(MetaboxBackend())
    reg.register(AnalyticalBackend())
    reg.register(S4Backend())
    return reg


__all__ = [
    "SimulationBackend", "BackendRegistry",
    "AnalyticalBackend", "MetaboxBackend", "S4Backend",
    "default_registry",
]