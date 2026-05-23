"""Optical performance analysis — PSF, MTF, Strehl, Zernike, EE, farfield.

Pure numpy. No backend dependency, no LLM, no MCP.
"""
from metaopticsai.analysis.psf import compute_psf
from metaopticsai.analysis.mtf import compute_mtf
from metaopticsai.analysis.strehl import compute_strehl_ratio
from metaopticsai.analysis.zernike import (
    zernike_polynomial, zernike_decomposition, ZERNIKE_NAMES,
)
from metaopticsai.analysis.encircled_energy import (
    compute_encircled_energy, radius_for_fraction,
)
from metaopticsai.analysis.farfield import compute_farfield

__all__ = [
    "compute_psf", "compute_mtf", "compute_strehl_ratio",
    "zernike_polynomial", "zernike_decomposition", "ZERNIKE_NAMES",
    "compute_encircled_energy", "radius_for_fraction",
    "compute_farfield",
]