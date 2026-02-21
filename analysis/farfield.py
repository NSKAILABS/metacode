"""
Far-field (Fraunhofer) diffraction pattern computation.

Computes the far-field intensity distribution after propagation
through the metalens phase mask.
"""

import numpy as np
from typing import Tuple, Optional


def compute_farfield(phase_mask: np.ndarray,
                     wavelength_nm: float,
                     pixel_size_um: float,
                     propagation_distance_um: float,
                     pad_factor: int = 4,
                     amplitude: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute far-field intensity via Fraunhofer diffraction.

    U(x',y') = FT{ A(x,y) * exp(jφ(x,y)) }

    evaluated at spatial frequencies fx = x'/(λz), fy = y'/(λz)

    Args:
        phase_mask: 2D phase array in radians
        wavelength_nm: Wavelength in nm
        pixel_size_um: Pixel pitch in um
        propagation_distance_um: Propagation distance z in um
        pad_factor: Zero-padding factor
        amplitude: Optional amplitude mask

    Returns:
        (intensity_2d, x_coords_um, y_coords_um)
    """
    N = phase_mask.shape[0]
    wl_um = wavelength_nm / 1000.0

    if amplitude is None:
        amplitude = np.ones_like(phase_mask)

    field = amplitude * np.exp(1j * phase_mask)

    N_pad = N * pad_factor
    padded = np.zeros((N_pad, N_pad), dtype=complex)
    start = (N_pad - N) // 2
    padded[start:start+N, start:start+N] = field

    # Fraunhofer propagation
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    intensity = np.abs(F) ** 2
    intensity /= intensity.max() if intensity.max() > 0 else 1.0

    # Output coordinates
    dx = pixel_size_um
    # Spatial frequency spacing: Δfx = 1/(N_pad * dx)
    # Output coordinate: x' = λz * fx
    coord_scale = wl_um * propagation_distance_um / (N_pad * dx)
    extent = N_pad * coord_scale / 2
    coords = np.linspace(-extent, extent, N_pad)

    return intensity, coords, coords


def compute_farfield_at_planes(phase_mask: np.ndarray,
                                wavelength_nm: float,
                                pixel_size_um: float,
                                z_values_um: list,
                                pad_factor: int = 2
                                ) -> list:
    """
    Compute far-field at multiple propagation distances.

    Args:
        phase_mask: 2D phase array
        wavelength_nm: Wavelength
        pixel_size_um: Pixel pitch
        z_values_um: List of propagation distances
        pad_factor: FFT padding

    Returns:
        List of (intensity, x, y) tuples
    """
    results = []
    for z in z_values_um:
        result = compute_farfield(phase_mask, wavelength_nm, pixel_size_um,
                                  z, pad_factor)
        results.append(result)
    return results