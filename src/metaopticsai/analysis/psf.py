"""Point Spread Function computation via Fraunhofer FFT.

    PSF = |FT{ A(x,y) · exp(j·φ(x,y)) }|²
"""
from __future__ import annotations

import numpy as np


def compute_psf(
    phase_mask: np.ndarray,
    wavelength_nm: float,
    pixel_size_um: float,
    focal_length_um: float = 0.0,
    pad_factor: int = 4,
    amplitude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (psf_2d, x_um, y_um) — normalized to peak 1."""
    N = phase_mask.shape[0]
    if amplitude is None:
        amplitude = np.ones_like(phase_mask)
    field = amplitude * np.exp(1j * phase_mask)

    N_pad = N * pad_factor
    padded = np.zeros((N_pad, N_pad), dtype=complex)
    s = (N_pad - N) // 2
    padded[s:s + N, s:s + N] = field

    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    psf = np.abs(F) ** 2
    peak = psf.max()
    if peak > 0:
        psf = psf / peak

    wl_um = wavelength_nm / 1000.0
    if focal_length_um > 0:
        coord_scale = wl_um * focal_length_um / (N_pad * pixel_size_um)
    else:
        coord_scale = wl_um / (N_pad * pixel_size_um)
    extent = N_pad * coord_scale / 2
    x = np.linspace(-extent, extent, N_pad)
    return psf, x, x.copy()