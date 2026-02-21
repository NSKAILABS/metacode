"""
Strehl Ratio computation.

Strehl = peak(PSF_aberrated) / peak(PSF_ideal)

A Strehl ratio of 1.0 indicates diffraction-limited performance.
Values > 0.8 are generally considered "well-corrected" (Maréchal criterion).
"""

import numpy as np
from typing import Tuple


def compute_strehl_ratio(phase_mask: np.ndarray,
                         pad_factor: int = 4
                         ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the Strehl ratio by comparing aberrated and ideal PSFs.

    Args:
        phase_mask: 2D phase array in radians (0 outside aperture)
        pad_factor: FFT zero-padding factor

    Returns:
        (strehl_ratio, psf_aberrated, psf_ideal)
    """
    N = phase_mask.shape[0]
    N_pad = N * pad_factor

    # Aperture mask (non-zero phase regions)
    aperture = (phase_mask != 0).astype(float)
    # If no clear aperture boundary, use full array
    if aperture.sum() == 0:
        aperture = np.ones_like(phase_mask)

    # Ideal field (flat phase, same aperture)
    field_ideal = aperture.astype(complex)
    padded_ideal = np.zeros((N_pad, N_pad), dtype=complex)
    start = (N_pad - N) // 2
    padded_ideal[start:start+N, start:start+N] = field_ideal

    F_ideal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded_ideal)))
    psf_ideal = np.abs(F_ideal) ** 2

    # Aberrated field
    field_aberr = aperture * np.exp(1j * phase_mask)
    padded_aberr = np.zeros((N_pad, N_pad), dtype=complex)
    padded_aberr[start:start+N, start:start+N] = field_aberr

    F_aberr = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded_aberr)))
    psf_aberr = np.abs(F_aberr) ** 2

    # Strehl ratio
    strehl = psf_aberr.max() / psf_ideal.max() if psf_ideal.max() > 0 else 0.0

    # Normalize for display
    psf_ideal /= psf_ideal.max() if psf_ideal.max() > 0 else 1.0
    psf_aberr /= psf_aberr.max() if psf_aberr.max() > 0 else 1.0

    return float(strehl), psf_aberr, psf_ideal