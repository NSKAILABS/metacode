"""
Point Spread Function (PSF) computation for metalens phase masks.

The PSF is computed via Fraunhofer diffraction:
    PSF = |FT{ A(x,y) * exp(j*φ(x,y)) }|²

where A is the aperture amplitude and φ is the phase mask.
"""

import numpy as np
from typing import Tuple, Optional


def compute_psf(phase_mask: np.ndarray,
                wavelength_nm: float,
                pixel_size_um: float,
                focal_length_um: float = 0.0,
                pad_factor: int = 4,
                amplitude: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the PSF of a metalens via 2D FFT.

    Args:
        phase_mask: 2D phase array in radians
        wavelength_nm: Design wavelength in nm
        pixel_size_um: Pixel pitch in um
        focal_length_um: Focal length (used for coordinate scaling)
        pad_factor: Zero-padding factor for FFT resolution
        amplitude: Optional amplitude mask (default: uniform)

    Returns:
        (psf_2d, x_um, y_um) — normalized PSF intensity and coordinate arrays in um
    """
    N = phase_mask.shape[0]

    # Aperture field
    if amplitude is None:
        amplitude = np.ones_like(phase_mask)

    # Apply circular mask if present (zero phase = outside aperture)
    field = amplitude * np.exp(1j * phase_mask)

    # Zero-pad for better resolution
    N_pad = N * pad_factor
    padded = np.zeros((N_pad, N_pad), dtype=complex)
    start = (N_pad - N) // 2
    padded[start:start+N, start:start+N] = field

    # 2D FFT → far-field
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    psf = np.abs(F) ** 2
    psf /= psf.max()  # Normalize to peak

    # Coordinate scaling
    wl_um = wavelength_nm / 1000.0
    dx = pixel_size_um  # sampling pitch
    # Angular frequency spacing: Δf = 1/(N_pad * dx)
    # At focal plane: x = λf * fx → Δx = λf/(N_pad * dx)
    if focal_length_um > 0:
        coord_scale = wl_um * focal_length_um / (N_pad * dx)
    else:
        coord_scale = wl_um / (N_pad * dx)  # dimensionless angular coords

    extent = N_pad * coord_scale / 2
    x = np.linspace(-extent, extent, N_pad)
    y = x.copy()

    return psf, x, y