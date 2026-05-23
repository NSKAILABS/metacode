"""Field propagation + scalar field metrics."""
from __future__ import annotations

import numpy as np

from metaopticsai.physics import propagation
from metaopticsai.physics.backends.base import BackendRegistry
from metaopticsai.tools.base import BaseTool
from metaopticsai.tools.schemas.analysis import FieldMetricInput
from metaopticsai.tools.schemas.propagation import PropagateInput


class PropagateFieldTool(BaseTool):
    name = "propagate_field"
    description = (
        "Propagate the optical field beyond the metasurface via the "
        "angular-spectrum method. Returns a handle to a 2-D intensity array."
    )
    Input = PropagateInput
    needs_store = True

    def __init__(self, store, backends: BackendRegistry):
        super().__init__(store=store)
        self.backends = backends

    def _run(self, p: PropagateInput):
        mask = self.store.get(p.mask_handle, expected_kind="phase_mask").payload
        result = propagation.propagate(
            phase_mask=mask,
            wavelength_nm=p.wavelength_nm,
            pixel_size_um=p.pixel_size_um,
            distance_um=p.propagation_distance_um,
            refractive_index=p.refractive_index,
        )
        n = result.intensity_2d.shape[0]
        x_um = np.arange(n) * p.pixel_size_um
        handle = self.store.put(
            "psf",
            {"psf_2d": result.intensity_2d, "x_um": x_um, "y_um": x_um},
            wavelength_nm=p.wavelength_nm,
            propagation_distance_um=p.propagation_distance_um,
        )
        return {
            "handle": handle,
            "shape": list(result.intensity_2d.shape),
            "peak_intensity": result.peak_intensity,
            "backend": result.backend,
        }


class ComputeMaxIntensityTool(BaseTool):
    name = "compute_max_intensity"
    description = "Maximum intensity of a stored PSF / propagated field. Higher = better-focused."
    Input = FieldMetricInput
    needs_store = True

    def _run(self, p: FieldMetricInput):
        psf = self.store.get(p.psf_handle, expected_kind="psf").payload["psf_2d"]
        return {"max_intensity": float(np.max(psf)), "shape": list(np.shape(psf))}


class ComputeCenterIntensityTool(BaseTool):
    name = "compute_center_intensity"
    description = "On-axis intensity at PSF center — useful as a differentiable FOM."
    Input = FieldMetricInput
    needs_store = True

    def _run(self, p: FieldMetricInput):
        psf = self.store.get(p.psf_handle, expected_kind="psf").payload["psf_2d"]
        cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
        return {"center_intensity": float(psf[cy, cx]), "center_pixel": [cx, cy]}


class ComputeMTFVolumeTool(BaseTool):
    name = "compute_mtf_volume"
    description = (
        "Volume under the normalized MTF surface of a stored PSF. "
        "Used internally by gradient-based Strehl optimization."
    )
    Input = FieldMetricInput
    needs_store = True

    def _run(self, p: FieldMetricInput):
        psf = self.store.get(p.psf_handle, expected_kind="psf").payload["psf_2d"]
        norm = psf / (psf.sum() if psf.sum() > 0 else 1.0)
        mtf = np.abs(np.fft.fftshift(np.fft.fft2(norm)))
        if mtf.max() > 0:
            mtf /= mtf.max()
        return {"mtf_volume": float(np.mean(mtf))}
