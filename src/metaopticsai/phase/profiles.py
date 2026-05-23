"""Phase profile generators for canonical metasurface elements.

Direct port of the old `core/phase_engine.py`, structured as free functions
rather than a class. Pure numpy; no backend dependency.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image


def fresnel_zone_lens(
    *,
    wavelength_nm: float,
    focal_length_um: float,
    diameter_um: float,
    pixel_size_um: float,
    circular: bool = True,
) -> np.ndarray:
    """Phase profile of a focusing FZL.

        φ(r) = 2π − (2π/λ)(√(r² + f²) − f)   mod 2π
    """
    wl_um = wavelength_nm / 1000.0
    R = diameter_um / 2.0
    nP = int(R / pixel_size_um)
    if nP < 1:
        raise ValueError("Diameter too small for the given pixel size.")

    N = 2 * nP
    x = np.linspace(-R, R, N)
    y = np.linspace(R, -R, N)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)

    phase = 2 * np.pi - (2 * np.pi / wl_um) * (np.sqrt(r ** 2 + focal_length_um ** 2) - focal_length_um)
    phase = phase % (2 * np.pi)

    if circular:
        phase[r > R] = 0.0
    return phase


def axicon(
    *,
    wavelength_nm: float,
    cone_angle_deg: float,
    diameter_um: float,
    pixel_size_um: float,
    circular: bool = True,
) -> np.ndarray:
    """Phase profile of an axicon (conical phase ramp, Bessel-beam generator).

        φ(r) = k · (R − r) · tan(α)   mod 2π
    """
    wl_um = wavelength_nm / 1000.0
    R = diameter_um / 2.0
    nP = int(R / pixel_size_um)
    if nP < 1:
        raise ValueError("Diameter too small for the given pixel size.")

    N = 2 * nP
    x = np.linspace(-R, R, N)
    y = np.linspace(R, -R, N)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)

    k = 2 * np.pi / wl_um
    alpha = np.radians(cone_angle_deg)
    phase = k * (R - r) * np.tan(alpha)
    phase = phase % (2 * np.pi)
    if circular:
        phase[r > R] = 0.0
    return phase


def spiral_phase_plate(
    *,
    charge: int,
    diameter_um: float,
    pixel_size_um: float,
    circular: bool = True,
) -> np.ndarray:
    """Spiral phase plate of topological charge `charge`.

        φ(θ) = l · θ   mod 2π
    """
    R = diameter_um / 2.0
    nP = int(R / pixel_size_um)
    if nP < 1:
        raise ValueError("Diameter too small for the given pixel size.")

    N = 2 * nP
    x = np.linspace(-R, R, N)
    y = np.linspace(R, -R, N)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    phase = charge * theta + charge * np.pi
    phase = phase % (2 * np.pi)
    if circular:
        phase[r > R] = 0.0
    return phase


def from_image(filepath: str) -> np.ndarray:
    """Load a grayscale image; map [0, 255] → [0, 2π)."""
    img = Image.open(filepath).convert("L")
    arr = np.array(img, dtype=float)
    return arr / 255.0 * 2 * np.pi