"""Strehl ratio — peak(PSF_aberrated) / peak(PSF_ideal).

Strehl > 0.8 is the Maréchal criterion for a well-corrected system.
"""
from __future__ import annotations

import numpy as np


def compute_strehl_ratio(
    phase_mask: np.ndarray,
    pad_factor: int = 4,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns (strehl_ratio, psf_aberrated_normalized, psf_ideal_normalized)."""
    N = phase_mask.shape[0]
    N_pad = N * pad_factor

    aperture = (phase_mask != 0).astype(float)
    if aperture.sum() == 0:
        aperture = np.ones_like(phase_mask)

    field_ideal = aperture.astype(complex)
    padded_ideal = np.zeros((N_pad, N_pad), dtype=complex)
    s = (N_pad - N) // 2
    padded_ideal[s:s + N, s:s + N] = field_ideal
    F_ideal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded_ideal)))
    psf_ideal = np.abs(F_ideal) ** 2

    field_aberr = aperture * np.exp(1j * phase_mask)
    padded_aberr = np.zeros((N_pad, N_pad), dtype=complex)
    padded_aberr[s:s + N, s:s + N] = field_aberr
    F_aberr = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded_aberr)))
    psf_aberr = np.abs(F_aberr) ** 2

    strehl = (psf_aberr.max() / psf_ideal.max()) if psf_ideal.max() > 0 else 0.0
    psf_ideal /= psf_ideal.max() if psf_ideal.max() > 0 else 1.0
    psf_aberr /= psf_aberr.max() if psf_aberr.max() > 0 else 1.0
    return float(strehl), psf_aberr, psf_ideal