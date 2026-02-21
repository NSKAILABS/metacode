"""
Encircled Energy computation.

EE(r) = ∫∫_{r'<r} PSF(r') dr' / ∫∫ PSF dr'

Measures what fraction of total energy is contained within radius r
of the PSF centroid. Key metric for imaging and focusing performance.
"""

import numpy as np
from typing import Tuple


def compute_encircled_energy(psf_2d: np.ndarray,
                             pixel_size_um: float = 1.0
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the encircled energy curve from a PSF.

    Args:
        psf_2d: 2D PSF intensity array
        pixel_size_um: Physical pixel size for radius scaling

    Returns:
        (radii_um, encircled_energy_fraction)
        where encircled_energy_fraction goes from 0 to ~1
    """
    N = psf_2d.shape[0]
    center = N // 2

    # Radial distance from PSF center (in pixels)
    y_idx, x_idx = np.ogrid[:N, :N]
    r = np.sqrt((x_idx - center)**2 + (y_idx - center)**2)

    total_energy = psf_2d.sum()
    if total_energy == 0:
        return np.array([0.0]), np.array([0.0])

    max_r = int(np.ceil(r.max()))
    max_r = min(max_r, center)

    radii = []
    ee = []
    cumulative = 0.0

    for ri in range(max_r):
        ring_mask = (r >= ri) & (r < ri + 1)
        cumulative += psf_2d[ring_mask].sum()
        radii.append((ri + 0.5) * pixel_size_um)
        ee.append(cumulative / total_energy)

    return np.array(radii), np.array(ee)


def compute_ee_at_radius(psf_2d: np.ndarray, target_fraction: float = 0.84
                         ) -> float:
    """
    Find the radius containing a given fraction of total energy.

    Args:
        psf_2d: 2D PSF
        target_fraction: Target encircled energy fraction (default 0.84 ≈ Airy disk first ring)

    Returns:
        Radius in pixels
    """
    radii, ee = compute_encircled_energy(psf_2d, pixel_size_um=1.0)
    idx = np.searchsorted(ee, target_fraction)
    if idx < len(radii):
        return radii[idx]
    return radii[-1]