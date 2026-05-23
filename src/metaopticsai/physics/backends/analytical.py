"""AnalyticalBackend — heuristic, TF-free FOM estimator.

Implements the same `SimulationBackend` interface as MetaboxBackend, but
without RCWA. Used for:

  - fast notebook iteration when TF is unavailable;
  - smoke tests in CI;
  - the inner loop of `optimize_metalens` when the FOM landscape only needs
    to be approximated.

The math is a direct port of the analytical block from the old
`core/automl.py::MetalensSimulator._simulate_analytical`.
"""
from __future__ import annotations

import time
import numpy as np

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.incidence import IncidenceSpec
from metaopticsai.domain.results import (
    FieldResult, RCWASweepResult, UnitCellResult,
)
from metaopticsai.physics.backends.base import SimulationBackend


_N_MATERIAL = {"TiO2": 2.35, "Si": 3.48, "GaN": 2.32, "SiN": 2.0}


class AnalyticalBackend(SimulationBackend):
    """Always-available, dependency-free heuristic backend."""

    name = "analytical"

    def is_available(self) -> bool:
        return True

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _phase_for_diameter(
        diam_nm: float, height_nm: float, n_mat: float, wl_nm: float,
    ) -> float:
        """First-order propagation phase for a pillar of given diameter.

        Uses a smoothly-varying scaling with diameter to keep gradients
        non-zero. This is intentionally heuristic — not a substitute for RCWA.
        """
        max_phase = 2.0 * np.pi * (n_mat - 1.0) * height_nm / wl_nm
        # Sigmoid-shaped fraction of max phase realized at a given diameter
        d_norm = (diam_nm - 50.0) / max(50.0, diam_nm)
        weight = 1.0 / (1.0 + np.exp(-4.0 * d_norm))
        return float(max_phase * weight)

    @staticmethod
    def _amplitude_for_diameter(
        diam_nm: float, height_nm: float, period_nm: float,
    ) -> float:
        """Heuristic transmission amplitude. Penalizes high aspect ratio and
        near-period diameters."""
        ar = height_nm / max(diam_nm, 1.0)
        ar_factor = 1.0 if ar <= 10 else max(0.0, 1.0 - (ar - 10) / 5.0)
        fill = diam_nm / max(period_nm, 1.0)
        fill_factor = 1.0 if fill <= 0.8 else max(0.0, 1.0 - (fill - 0.8) * 3.0)
        return float(0.7 + 0.3 * ar_factor * fill_factor)

    # ── interface ────────────────────────────────────────────────────────

    def simulate_unit_cell(self, params, incidence):
        t0 = time.perf_counter()
        n_mat = _N_MATERIAL.get(params.pillar_material, 2.0)
        wl_nm = incidence.wavelength_m[0] * 1e9
        d = params.mean_diameter_nm
        phase = self._phase_for_diameter(d, params.pillar_height_nm, n_mat, wl_nm)
        amp = self._amplitude_for_diameter(d, params.pillar_height_nm, params.periodicity_nm)
        tx = amp * np.exp(1j * phase)
        return UnitCellResult(
            tx_complex=complex(tx),
            ty_complex=complex(tx),
            amplitude=amp,
            phase_rad=phase,
            elapsed_s=time.perf_counter() - t0,
            backend=self.name,
        )

    def sweep_unit_cells(self, params, incidence, sweep_param, sweep_values):
        if sweep_param != "diameter":
            raise NotImplementedError(
                f"AnalyticalBackend only supports diameter sweeps, got {sweep_param!r}"
            )
        t0 = time.perf_counter()
        n_mat = _N_MATERIAL.get(params.pillar_material, 2.0)
        wl_nm = incidence.wavelength_m[0] * 1e9
        vals = np.asarray(sweep_values, dtype=float)

        phases = np.array([
            self._phase_for_diameter(d, params.pillar_height_nm, n_mat, wl_nm)
            for d in vals
        ])
        amps = np.array([
            self._amplitude_for_diameter(d, params.pillar_height_nm, params.periodicity_nm)
            for d in vals
        ])
        phases = phases - phases[0]  # zero-align
        return RCWASweepResult(
            sweep_param=sweep_param,
            sweep_values=vals,
            amplitudes=amps,
            phases_rad=phases,
            wavelength_m=incidence.wavelength_m[0],
            phase_coverage=float((phases.max() - phases.min()) / (2 * np.pi)),
            mean_transmission=float(amps.mean()),
            elapsed_s=time.perf_counter() - t0,
            backend=self.name,
        )

    def propagate_field(
        self, phase_mask, wavelength_m, pixel_size_m, distance_m,
        refractive_index=1.0,
    ):
        """Fraunhofer (FFT) propagation — adequate for far-field PSF estimate."""
        n = phase_mask.shape[0]
        field = np.exp(1j * phase_mask)
        pad = n * 4
        padded = np.zeros((pad, pad), dtype=complex)
        s = (pad - n) // 2
        padded[s:s + n, s:s + n] = field
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
        intensity = np.abs(F) ** 2
        peak = float(intensity.max())
        normed = intensity / max(peak, 1e-30)
        return FieldResult(
            intensity_2d=normed,
            peak_intensity=peak,
            shape=tuple(normed.shape),  # type: ignore[arg-type]
            wavelength_m=wavelength_m,
            backend=self.name,
        )