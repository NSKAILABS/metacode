"""Encircled-energy curve from a PSF."""
from __future__ import annotations

import numpy as np


def compute_encircled_energy(
    psf_2d: np.ndarray,
    pixel_size_um: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (radii_um, ee_fraction)."""
    N = psf_2d.shape[0]
    center = N // 2
    y_idx, x_idx = np.ogrid[:N, :N]
    r = np.sqrt((x_idx - center) ** 2 + (y_idx - center) ** 2)
    total = psf_2d.sum()
    if total == 0:
        return np.array([0.0]), np.array([0.0])

    max_r = min(int(np.ceil(r.max())), center)
    radii, ee = [], []
    cumulative = 0.0
    for ri in range(max_r):
        ring_mask = (r >= ri) & (r < ri + 1)
        cumulative += psf_2d[ring_mask].sum()
        radii.append((ri + 0.5) * pixel_size_um)
        ee.append(cumulative / total)
    return np.array(radii), np.array(ee)


def radius_for_fraction(psf_2d: np.ndarray, target: float = 0.84) -> float:
    """Radius (in pixels) containing `target` of total energy."""
    radii, ee = compute_encircled_energy(psf_2d)
    idx = int(np.searchsorted(ee, target))
    return float(radii[idx] if idx < len(radii) else radii[-1])