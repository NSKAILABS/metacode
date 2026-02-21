"""
Modulation Transfer Function (MTF) computation.

MTF = |FT{PSF}| normalized so MTF(0) = 1.
Also computes the diffraction-limited MTF for comparison.
"""

import numpy as np
from typing import Tuple


def compute_mtf(psf_2d: np.ndarray,
                pixel_size_um: float,
                wavelength_nm: float,
                focal_length_um: float = 0.0,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the radially-averaged MTF from a PSF.

    Args:
        psf_2d: 2D PSF intensity array (normalized)
        pixel_size_um: Pixel pitch in um
        wavelength_nm: Wavelength in nm
        focal_length_um: Focal length in um (for cutoff frequency)

    Returns:
        (frequencies_lp_mm, mtf_values, mtf_diffraction_limit)
    """
    N = psf_2d.shape[0]

    # OTF = FT of PSF
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf_2d)))
    mtf_2d = np.abs(otf)
    mtf_2d /= mtf_2d.max()

    # Radial average
    center = N // 2
    y_idx, x_idx = np.ogrid[:N, :N]
    r = np.sqrt((x_idx - center)**2 + (y_idx - center)**2).astype(int)

    max_r = min(center, N - center - 1)
    radial_mtf = np.zeros(max_r)
    for ri in range(max_r):
        ring = mtf_2d[r == ri]
        if len(ring) > 0:
            radial_mtf[ri] = ring.mean()

    # Frequency axis (cycles per mm)
    # freq spacing = 1/(N * pixel_size_um) in cycles/um → *1000 for cycles/mm
    df = 1.0 / (N * pixel_size_um) * 1000.0  # lp/mm
    freqs = np.arange(max_r) * df

    # Diffraction-limited MTF (circular aperture)
    # Cutoff frequency: f_c = D/(λf) for a lens, or 1/(λ F/#)
    wl_um = wavelength_nm / 1000.0
    if focal_length_um > 0:
        # Estimate NA from mask size
        # This is approximate — assumes the mask fills the aperture
        mask_radius_um = N * pixel_size_um / 2.0
        NA = mask_radius_um / np.sqrt(mask_radius_um**2 + focal_length_um**2)
        f_cutoff = 2 * NA / wl_um * 1000.0  # lp/mm
    else:
        f_cutoff = freqs[-1] if len(freqs) > 0 else 1.0

    # Diffraction-limited MTF for circular aperture:
    # MTF(f) = (2/π)[arccos(f/fc) - (f/fc)√(1-(f/fc)²)]
    f_norm = freqs / f_cutoff
    f_norm = np.clip(f_norm, 0, 1)
    mtf_diff = (2 / np.pi) * (np.arccos(f_norm) - f_norm * np.sqrt(1 - f_norm**2))
    mtf_diff[f_norm >= 1.0] = 0.0

    return freqs, radial_mtf, mtf_diff


def compute_mtf_cross_section(psf_2d: np.ndarray,
                               pixel_size_um: float,
                               axis: str = 'x'
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D MTF along a specific axis through PSF center.

    Args:
        psf_2d: 2D PSF array
        pixel_size_um: Pixel pitch in um
        axis: 'x' or 'y'

    Returns:
        (frequencies_lp_mm, mtf_1d)
    """
    N = psf_2d.shape[0]
    center = N // 2

    if axis == 'x':
        line = psf_2d[center, :]
    else:
        line = psf_2d[:, center]

    otf_1d = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(line)))
    mtf_1d = np.abs(otf_1d)
    mtf_1d /= mtf_1d.max()

    df = 1.0 / (N * pixel_size_um) * 1000.0
    freqs = np.arange(N) * df
    # Take positive half
    half = N // 2
    return freqs[:half], mtf_1d[:half]