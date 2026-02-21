"""
Zernike Polynomial Decomposition.

Decomposes a phase mask into Zernike polynomial coefficients to
identify specific optical aberrations (defocus, astigmatism, coma,
spherical aberration, etc.).

Uses the Noll indexing convention for Zernike polynomials.
"""

import numpy as np
import math
from typing import Dict, List, Tuple
from functools import lru_cache


# Noll-indexed Zernike names
ZERNIKE_NAMES = {
    1: "Piston",
    2: "Tilt (x)",
    3: "Tilt (y)",
    4: "Defocus",
    5: "Astigmatism (oblique)",
    6: "Astigmatism (vertical)",
    7: "Coma (vertical)",
    8: "Coma (horizontal)",
    9: "Trefoil (vertical)",
    10: "Trefoil (oblique)",
    11: "Primary Spherical",
    12: "Secondary Astigmatism (v)",
    13: "Secondary Astigmatism (o)",
    14: "Secondary Coma (v)",
    15: "Secondary Coma (h)",
}


def _noll_to_nm(j: int) -> Tuple[int, int]:
    """Convert Noll index j to (n, m) radial and azimuthal orders."""
    n = 0
    j_count = 0
    while j_count < j:
        n += 1
        j_count += n
    # Now n is the radial order
    n -= 1
    # Recalculate
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * (j - 1))) / 2))
    m_options = list(range(-n, n + 1, 2))
    idx = j - n * (n + 1) // 2 - 1
    if idx < len(m_options):
        m = m_options[idx]
    else:
        m = 0
    return n, m


def _noll_indices(j: int) -> Tuple[int, int]:
    """More robust Noll index to (n, m) conversion."""
    # Standard algorithm
    n = 0
    j1 = j - 1
    while j1 >= n + 1:
        n += 1
        j1 -= n
    m = -n + 2 * j1
    if j % 2 == 0:
        m = abs(m)
    else:
        if m != 0:
            m = -abs(m)
    # Fix sign convention
    n_calc = int((-1 + np.sqrt(1 + 8 * (j - 1))) / 2)
    if n_calc * (n_calc + 1) // 2 >= j:
        n_calc -= 1
    n = n_calc + 1 if (n_calc + 1) * (n_calc + 2) // 2 >= j else n_calc

    # Simple table for first 15
    table = {
        1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0),
        5: (2, -2), 6: (2, 2), 7: (3, -1), 8: (3, 1),
        9: (3, -3), 10: (3, 3), 11: (4, 0), 12: (4, 2),
        13: (4, -2), 14: (5, 1), 15: (5, -1),
    }
    if j in table:
        return table[j]
    return (n, 0)


def _radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Compute radial Zernike polynomial R_n^m(ρ)."""
    m_abs = abs(m)
    result = np.zeros_like(rho)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1) ** s * math.factorial(n - s) /
                 (math.factorial(s) *
                  math.factorial((n + m_abs) // 2 - s) *
                  math.factorial((n - m_abs) // 2 - s)))
        result += coeff * rho ** (n - 2 * s)
    return result


def zernike_polynomial(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute Zernike polynomial Z_j(ρ,θ) using Noll indexing.

    Args:
        j: Noll index (1-based)
        rho: Radial coordinate array (0 to 1)
        theta: Azimuthal angle array (0 to 2π)

    Returns:
        2D Zernike polynomial values
    """
    n, m = _noll_indices(j)
    R = _radial_polynomial(n, m, rho)

    if m == 0:
        norm = np.sqrt(n + 1)
        return norm * R
    elif m > 0:
        norm = np.sqrt(2 * (n + 1))
        return norm * R * np.cos(m * theta)
    else:
        norm = np.sqrt(2 * (n + 1))
        return norm * R * np.sin(abs(m) * theta)


def zernike_decomposition(phase_mask: np.ndarray,
                          n_terms: int = 15
                          ) -> Tuple[Dict[int, float], Dict[int, str]]:
    """
    Decompose a phase mask into Zernike polynomial coefficients.

    Args:
        phase_mask: 2D phase array in radians
        n_terms: Number of Zernike terms to compute (default 15)

    Returns:
        (coefficients_dict, names_dict)
        coefficients_dict: {noll_index: coefficient_value}
        names_dict: {noll_index: aberration_name}
    """
    N = phase_mask.shape[0]
    center = N // 2

    # Create normalized polar coordinates
    y, x = np.ogrid[:N, :N]
    x_c = (np.arange(N) - center) / center
    y_c = (np.arange(N) - center) / center
    X, Y = np.meshgrid(x_c, y_c)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Unit circle mask
    mask = rho <= 1.0
    n_pixels = mask.sum()

    if n_pixels == 0:
        return {}, {}

    # Extract phase within aperture (subtract piston)
    phase_in = phase_mask[mask]

    coefficients = {}
    names = {}

    for j in range(1, n_terms + 1):
        Z_j = zernike_polynomial(j, rho, theta)
        # Inner product: c_j = (1/N) * Σ φ * Z_j over unit disk
        c_j = np.sum(phase_mask[mask] * Z_j[mask]) / n_pixels
        coefficients[j] = float(c_j)
        names[j] = ZERNIKE_NAMES.get(j, f"Z_{j}")

    return coefficients, names