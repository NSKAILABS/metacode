"""Phase quantization helpers — continuous radians → discrete levels."""
from __future__ import annotations

import numpy as np


def quantize(
    phase: np.ndarray,
    n_levels: int,
    *,
    phase_min_deg: float = 0.0,
    phase_max_deg: float = 360.0,
) -> tuple[np.ndarray, list[float]]:
    """Quantize a continuous phase mask (in radians) into integer levels.

    Returns:
        quantized: 2D int array with values in [1, n_levels]
        bins: bin-edge centers in degrees, length n_levels
    """
    phase_deg = np.degrees(phase)
    phase_range = phase_max_deg - phase_min_deg
    interval = phase_range / n_levels
    bins = [phase_min_deg + i * interval for i in range(n_levels)]

    p_min, p_max = float(phase_deg.min()), float(phase_deg.max())
    if p_max > p_min:
        normalized = (phase_deg - p_min) / (p_max - p_min) * phase_range + phase_min_deg
    else:
        normalized = np.full_like(phase_deg, phase_min_deg)

    quantized = np.digitize(normalized, bins)
    quantized = np.clip(quantized, 1, n_levels)
    return quantized, bins