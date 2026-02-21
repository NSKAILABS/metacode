"""
Phase mask generation engine.

Generates phase profiles for:
  - Fresnel Zone Lens (FZL)
  - Axicon
  - Spiral Phase Plate (SPP)
  - Custom imported images
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional


class PhaseEngine:
    """Generates 2D phase masks for various optical elements."""

    @staticmethod
    def fresnel_zone_lens(wavelength_nm: float, focal_length_um: float,
                          diameter_um: float, pixel_size_um: float,
                          circular: bool = True) -> np.ndarray:
        """
        Generate FZL (converging lens) phase mask.

        Phase(r) = 2π - (2π/λ)(√(r² + f²) - f)  mod 2π

        Args:
            wavelength_nm: Design wavelength in nm
            focal_length_um: Focal length in micrometers
            diameter_um: Lens diameter in micrometers
            pixel_size_um: Pixel pitch in micrometers
            circular: Apply circular aperture

        Returns:
            2D numpy array of phase values in radians [0, 2π)
        """
        wl_um = wavelength_nm / 1000.0
        R = diameter_um / 2.0
        nP = int(R / pixel_size_um)
        if nP < 1:
            raise ValueError("Diameter too small for given pixel size.")

        N = 2 * nP
        x = np.linspace(-R, R, N)
        y = np.linspace(R, -R, N)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2)

        phase = 2 * np.pi - (2 * np.pi / wl_um) * (np.sqrt(r**2 + focal_length_um**2) - focal_length_um)
        phase = phase % (2 * np.pi)

        if circular:
            mask = r > R
            phase[mask] = 0.0

        return phase

    @staticmethod
    def axicon(wavelength_nm: float, cone_angle_deg: float,
               diameter_um: float, pixel_size_um: float,
               circular: bool = True) -> np.ndarray:
        """
        Generate Axicon phase mask.

        Phase(r) = k * (R - r) * tan(α)  mod 2π

        Args:
            wavelength_nm: Design wavelength in nm
            cone_angle_deg: Half-cone angle in degrees
            diameter_um: Diameter in micrometers
            pixel_size_um: Pixel pitch in micrometers
            circular: Apply circular aperture

        Returns:
            2D phase array in radians [0, 2π)
        """
        wl_um = wavelength_nm / 1000.0
        R = diameter_um / 2.0
        nP = int(R / pixel_size_um)
        if nP < 1:
            raise ValueError("Diameter too small for given pixel size.")

        N = 2 * nP
        x = np.linspace(-R, R, N)
        y = np.linspace(R, -R, N)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2)

        k = 2 * np.pi / wl_um
        alpha = np.radians(cone_angle_deg)
        phase = k * (R - r) * np.tan(alpha)
        phase = phase % (2 * np.pi)

        if circular:
            phase[r > R] = 0.0

        return phase

    @staticmethod
    def spiral_phase_plate(wavelength_nm: float, charge: int,
                           diameter_um: float, pixel_size_um: float,
                           circular: bool = True) -> np.ndarray:
        """
        Generate Spiral Phase Plate (vortex beam generator).

        Phase(θ) = l * θ  mod 2π, where l is the topological charge.

        Args:
            wavelength_nm: Design wavelength (not used in formula but kept for consistency)
            charge: Topological charge (integer)
            diameter_um: Diameter in micrometers
            pixel_size_um: Pixel pitch in micrometers
            circular: Apply circular aperture

        Returns:
            2D phase array in radians [0, 2π)
        """
        R = diameter_um / 2.0
        nP = int(R / pixel_size_um)
        if nP < 1:
            raise ValueError("Diameter too small for given pixel size.")

        N = 2 * nP
        x = np.linspace(-R, R, N)
        y = np.linspace(R, -R, N)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2)

        theta = np.arctan2(Y, X)
        phase = charge * theta + charge * np.pi
        phase = phase % (2 * np.pi)

        if circular:
            phase[r > R] = 0.0

        return phase

    @staticmethod
    def from_image(filepath: str) -> np.ndarray:
        """
        Load a grayscale image and map pixel intensity [0,255] → phase [0, 2π).

        Args:
            filepath: Path to image file (PNG, JPG, BMP)

        Returns:
            2D phase array in radians
        """
        img = Image.open(filepath).convert('L')
        arr = np.array(img, dtype=float)
        phase = arr / 255.0 * 2 * np.pi
        return phase

    @staticmethod
    def quantize_phase(phase: np.ndarray, n_levels: int,
                       phase_min_deg: float = 0.0,
                       phase_max_deg: float = 360.0) -> Tuple[np.ndarray, list]:
        """
        Quantize continuous phase into discrete levels.

        Args:
            phase: 2D array of phase in radians
            n_levels: Number of quantization levels
            phase_min_deg: Minimum phase in degrees
            phase_max_deg: Maximum phase in degrees

        Returns:
            (quantized_indices, bin_edges_degrees)
        """
        phase_deg = np.degrees(phase)
        phase_range = phase_max_deg - phase_min_deg
        interval = phase_range / n_levels

        bins = [phase_min_deg + i * interval for i in range(n_levels)]

        # Normalize to [phase_min, phase_max]
        p_min, p_max = phase_deg.min(), phase_deg.max()
        if p_max > p_min:
            normalized = (phase_deg - p_min) / (p_max - p_min) * phase_range + phase_min_deg
        else:
            normalized = np.full_like(phase_deg, phase_min_deg)

        quantized = np.digitize(normalized, bins)
        quantized = np.clip(quantized, 1, n_levels)

        return quantized, bins