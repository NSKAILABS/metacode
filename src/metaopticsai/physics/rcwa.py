"""High-level RCWA facade. The standard entry point for code that wants to
run a sweep without thinking about which backend is wired in.

Use it like:

    from metaopticsai.physics.rcwa import sweep_diameter
    result = sweep_diameter(params, wavelength_nm=532, n=20)

For more control, instantiate a `BackendRegistry` and call backend methods
directly.
"""
from __future__ import annotations

import numpy as np

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.incidence import IncidenceSpec
from metaopticsai.domain.results import RCWASweepResult
from metaopticsai.physics.backends import default_registry
from metaopticsai.physics.backends.base import BackendRegistry


# Module-level shared registry. Composition roots may replace this.
_REGISTRY: BackendRegistry = default_registry()


def get_registry() -> BackendRegistry:
    return _REGISTRY


def set_registry(reg: BackendRegistry) -> None:
    """Inject a custom registry — used by tests."""
    global _REGISTRY
    _REGISTRY = reg


def sweep_diameter(
    params: MetalensDesignParams,
    *,
    wavelength_nm: float | None = None,
    n: int = 20,
    backend: str = "auto",
) -> RCWASweepResult:
    """Sweep pillar diameter from `params.min_diameter_nm` → `params.max_diameter_nm`."""
    wl_nm = wavelength_nm if wavelength_nm is not None else params.wavelength_nm
    incidence = IncidenceSpec(wavelength_m=[wl_nm * 1e-9])
    vals = np.linspace(params.min_diameter_nm, params.max_diameter_nm, n)
    return _REGISTRY.get(backend).sweep_unit_cells(
        params, incidence, sweep_param="diameter", sweep_values=vals,
    )