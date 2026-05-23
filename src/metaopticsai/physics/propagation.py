"""Field propagation facade. Uses the registry's preferred backend."""
from __future__ import annotations

import numpy as np

from metaopticsai.domain.results import FieldResult
from metaopticsai.physics.rcwa import get_registry


def propagate(
    phase_mask: np.ndarray,
    *,
    wavelength_nm: float,
    pixel_size_um: float,
    distance_um: float,
    refractive_index: float = 1.0,
    backend: str = "auto",
) -> FieldResult:
    """Angular-spectrum propagation through `distance_um`.

    All inputs in their natural units (nm, µm). Conversion to SI happens here.
    """
    return get_registry().get(backend).propagate_field(
        phase_mask=phase_mask,
        wavelength_m=wavelength_nm * 1e-9,
        pixel_size_m=pixel_size_um * 1e-6,
        distance_m=distance_um * 1e-6,
        refractive_index=refractive_index,
    )