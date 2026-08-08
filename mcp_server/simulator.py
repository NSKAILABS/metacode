"""Thin wrapper around the *existing* verified metabox3-backed physics.

We reuse `metaopticsai.physics.rcwa.sweep_diameter`, which in turn uses
the `metabox3` vendor backend when TensorFlow is installed, and falls
back to the always-available analytical backend otherwise. No metabox3
API is re-invented here.
"""
from __future__ import annotations

from typing import Any

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.physics.rcwa import sweep_diameter


def run_diameter_sweep(
    params: dict[str, Any],
    n_samples: int = 20,
    backend: str = "auto",
) -> dict[str, Any]:
    """Validate `params` through Pydantic, then dispatch to the RCWA facade.

    Returns a JSON-serialisable dict (NumPy arrays are converted to lists).
    """
    design = MetalensDesignParams(**params)
    result = sweep_diameter(
        design,
        wavelength_nm=design.wavelength_nm,
        n=n_samples,
        backend=backend,
    )
    return {
        "backend": result.backend,
        "sweep_param": result.sweep_param,
        "wavelength_nm": design.wavelength_nm,
        "n_samples": len(result.sweep_values),
        "sweep_values_nm": result.sweep_values.tolist(),
        "amplitudes": result.amplitudes.tolist(),
        "phases_rad": result.phases_rad.tolist(),
        "phase_coverage": float(result.phase_coverage),
        "mean_transmission": float(result.mean_transmission),
        "elapsed_s": float(result.elapsed_s),
    }


def available_backends() -> list[str]:
    """Introspection helper — surfaces which backends can actually run."""
    from metaopticsai.physics.rcwa import get_registry
    reg = get_registry()
    return reg.available()
