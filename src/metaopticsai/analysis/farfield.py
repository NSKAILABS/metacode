"""Fraunhofer far-field intensity propagation."""
from __future__ import annotations

import numpy as np


def compute_farfield(
    phase_mask: np.ndarray,
    wavelength_nm: float,
    pixel_size_um: float,
    propagation_distance_um: float,
    pad_factor: int = 4,
    amplitude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (intensity_2d, x_um, y_um)."""
    N = phase_mask.shape[0]
    wl_um = wavelength_nm / 1000.0
    if amplitude is None:
        amplitude = np.ones_like(phase_mask)
    field = amplitude * np.exp(1j * phase_mask)

    N_pad = N * pad_factor
    padded = np.zeros((N_pad, N_pad), dtype=complex)
    s = (N_pad - N) // 2
    padded[s:s + N, s:s + N] = field
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    intensity = np.abs(F) ** 2
    if intensity.max() > 0:
        intensity /= intensity.max()

    coord_scale = wl_um * propagation_distance_um / (N_pad * pixel_size_um)
    extent = N_pad * coord_scale / 2
    coords = np.linspace(-extent, extent, N_pad)
    return intensity, coords, coords.copy()