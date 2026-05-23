"""Zernike polynomial decomposition (Noll indexing)."""
from __future__ import annotations

import math
import numpy as np


ZERNIKE_NAMES = {
    1: "Piston", 2: "Tilt (x)", 3: "Tilt (y)", 4: "Defocus",
    5: "Astigmatism (oblique)", 6: "Astigmatism (vertical)",
    7: "Coma (vertical)", 8: "Coma (horizontal)",
    9: "Trefoil (vertical)", 10: "Trefoil (oblique)",
    11: "Primary Spherical",
    12: "Secondary Astigmatism (v)", 13: "Secondary Astigmatism (o)",
    14: "Secondary Coma (v)", 15: "Secondary Coma (h)",
}


_NOLL_NM = {
    1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0),
    5: (2, -2), 6: (2, 2), 7: (3, -1), 8: (3, 1),
    9: (3, -3), 10: (3, 3), 11: (4, 0), 12: (4, 2),
    13: (4, -2), 14: (5, 1), 15: (5, -1),
}


def _radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
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
    """Zernike polynomial Z_j(ρ,θ) (Noll-indexed)."""
    n, m = _NOLL_NM.get(j, (0, 0))
    R = _radial_polynomial(n, m, rho)
    if m == 0:
        norm = np.sqrt(n + 1)
        return norm * R
    if m > 0:
        return np.sqrt(2 * (n + 1)) * R * np.cos(m * theta)
    return np.sqrt(2 * (n + 1)) * R * np.sin(abs(m) * theta)


def zernike_decomposition(
    phase_mask: np.ndarray,
    n_terms: int = 15,
) -> tuple[dict[int, float], dict[int, str]]:
    """Decompose a phase mask into Zernike coefficients.

    Returns:
        coefficients: {Noll index → coefficient (radians)}
        names: {Noll index → aberration name}
    """
    N = phase_mask.shape[0]
    center = N // 2
    x_c = (np.arange(N) - center) / center
    X, Y = np.meshgrid(x_c, x_c)
    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)
    mask = rho <= 1.0
    n_pixels = mask.sum()
    if n_pixels == 0:
        return {}, {}

    coefficients: dict[int, float] = {}
    names: dict[int, str] = {}
    for j in range(1, n_terms + 1):
        Z_j = zernike_polynomial(j, rho, theta)
        c_j = np.sum(phase_mask[mask] * Z_j[mask]) / n_pixels
        coefficients[j] = float(c_j)
        names[j] = ZERNIKE_NAMES.get(j, f"Z_{j}")

    return coefficients, names