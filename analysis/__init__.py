"""
Analysis Module — Optical Performance Metrics
===============================================

Contains all optical analysis functions:
- PSF (Point Spread Function)
- MTF (Modulation Transfer Function)
- Strehl ratio
- Encircled energy
- Zernike decomposition
- Far-field propagation
"""

from analysis.psf import compute_psf
from analysis.mtf import compute_mtf
from analysis.strehl import compute_strehl_ratio
from analysis.encircled_energy import compute_encircled_energy
from analysis.zernike import zernike_polynomial, zernike_decomposition
from analysis.farfield import compute_farfield

__all__ = [
    'compute_psf',
    'compute_mtf',
    'compute_strehl_ratio',
    'compute_encircled_energy',
    'zernike_polynomial',
    'zernike_decomposition',
    'compute_farfield',
]
