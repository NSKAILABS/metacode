"""Modulation Transfer Function — radial average + diffraction-limit reference."""
from __future__ import annotations

import numpy as np


def compute_mtf(
    psf_2d: np.ndarray,
    pixel_size_um: float,
    wavelength_nm: float,
    focal_length_um: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (freqs_lpmm, mtf_radial, mtf_diff_limit)."""
    N = psf_2d.shape[0]
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf_2d)))
    mtf_2d = np.abs(otf)
    if mtf_2d.max() > 0:
        mtf_2d = mtf_2d / mtf_2d.max()

    center = N // 2
    y_idx, x_idx = np.ogrid[:N, :N]
    r = np.sqrt((x_idx - center) ** 2 + (y_idx - center) ** 2).astype(int)

    max_r = min(center, N - center - 1)
    radial_mtf = np.zeros(max_r)
    for ri in range(max_r):
        ring = mtf_2d[r == ri]
        if len(ring) > 0:
            radial_mtf[ri] = ring.mean()

    df = 1.0 / (N * pixel_size_um) * 1000.0  # lp/mm
    freqs = np.arange(max_r) * df

    wl_um = wavelength_nm / 1000.0
    if focal_length_um > 0:
        mask_radius_um = N * pixel_size_um / 2.0
        NA = mask_radius_um / np.sqrt(mask_radius_um ** 2 + focal_length_um ** 2)
        f_cutoff = 2 * NA / wl_um * 1000.0
    else:
        f_cutoff = freqs[-1] if len(freqs) > 0 else 1.0

    f_norm = np.clip(freqs / f_cutoff, 0, 1)
    mtf_diff = (2 / np.pi) * (np.arccos(f_norm) - f_norm * np.sqrt(1 - f_norm ** 2))
    mtf_diff[f_norm >= 1.0] = 0.0

    return freqs, radial_mtf, mtf_diff