"""Result types returned by physics backends.

These are *plain dataclasses* (not Pydantic) because they hold numpy arrays
that Pydantic v2 can't validate efficiently. They cross the backend boundary
in one direction: backend → domain → tool. No TF/metabox types ever appear in
fields.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass
class UnitCellResult:
    """Single-cell RCWA result."""
    tx_complex: complex
    ty_complex: complex
    amplitude: float
    phase_rad: float
    elapsed_s: float
    backend: str


@dataclasses.dataclass
class RCWASweepResult:
    """Result of a parametric sweep of unit cells."""
    sweep_param: str
    sweep_values: np.ndarray            # (N,) the swept axis
    amplitudes: np.ndarray              # (N,) |t|
    phases_rad: np.ndarray              # (N,) unwrapped, zero-aligned
    wavelength_m: float
    phase_coverage: float               # fraction of 2π
    mean_transmission: float
    elapsed_s: float
    backend: str

    def to_payload(self) -> dict[str, Any]:
        """Serialize for handle store. Arrays → lists."""
        return {
            "sweep_param": self.sweep_param,
            "sweep_values": self.sweep_values.tolist(),
            "amplitudes": self.amplitudes.tolist(),
            "phases_rad": self.phases_rad.tolist(),
            "wavelength_nm": self.wavelength_m * 1e9,
            "phase_coverage": self.phase_coverage,
            "mean_transmission": self.mean_transmission,
            "backend": self.backend,
        }


@dataclasses.dataclass
class FieldResult:
    """Propagated optical field intensity."""
    intensity_2d: np.ndarray
    peak_intensity: float
    shape: tuple[int, int]
    wavelength_m: float
    backend: str

    def to_payload(self, x_um: np.ndarray | None = None,
                   y_um: np.ndarray | None = None) -> dict[str, Any]:
        return {
            "psf_2d": self.intensity_2d,
            "x_um": (x_um.tolist() if x_um is not None
                     else list(range(self.intensity_2d.shape[1]))),
            "y_um": (y_um.tolist() if y_um is not None
                     else list(range(self.intensity_2d.shape[0]))),
            "peak_intensity": self.peak_intensity,
        }


@dataclasses.dataclass
class PSFResult:
    """Point-spread function + derived metrics."""
    psf_2d: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    strehl_ratio: float
    fwhm_um: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "psf_2d": self.psf_2d,
            "x_um": self.x_um,
            "y_um": self.y_um,
            "strehl_ratio": self.strehl_ratio,
            "fwhm_um": self.fwhm_um,
        }


@dataclasses.dataclass
class FOMResult:
    """A scalar figure-of-merit with provenance.

    Used by optimizers — they consume an FOMResult per evaluation rather than
    a raw float, so the source of the score is always traceable.
    """
    value: float
    name: str                           # "strehl_ratio", "max_intensity", …
    backend: str
    iteration: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)