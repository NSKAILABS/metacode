"""DesignState — mutable container tracking the entire design pipeline.

Migrated from the old `core/design.py`. This is a working scratchpad used by
notebooks and the (legacy) GUI workflow. The Self-RAG LangGraph workflow uses
a TypedDict in `workflows/state.py` instead.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class DesignState:
    """Mutable state object tracking the entire design pipeline.

    Holds wavelength, FDTD entries, phase mask, quantization bins, library
    geometry parameters, and output paths.
    """

    # ── Wavelength & Material ──────────────────────────────────────────────
    wavelength: int = 0                 # nm
    shape: str = "Cylinder"             # 'Cylinder' | 'Cross' | 'Fin'
    material: str = ""
    height: float = 0.0                 # nm
    period: float = 0.0                 # nm

    # ── FDTD reference data ────────────────────────────────────────────────
    dimensions: Optional[list[float]] = None
    phases: Optional[list[float]] = None
    transmissions: Optional[list[float]] = None
    is_user_data: bool = False

    # ── Cross / Fin params ─────────────────────────────────────────────────
    cross_width: float = 0.0
    fin_width: float = 0.0
    fin_length: float = 0.0

    # ── Phase mask design ──────────────────────────────────────────────────
    phase_mask: Optional[np.ndarray] = None      # continuous (radians)
    quantized_mask: Optional[np.ndarray] = None  # integer labels
    bins: Optional[list[float]] = None

    # ── Design parameters ──────────────────────────────────────────────────
    unitcells_per_pixel: int = 1
    phase_levels: int = 8
    phase_min: float = 0.0               # degrees
    phase_max: float = 360.0             # degrees
    is_circular: bool = True

    # ── Library element params ─────────────────────────────────────────────
    fzl_focal_length: float = 0.0        # um
    fzl_diameter: float = 0.0            # um
    axicon_angle: float = 0.0            # degrees
    axicon_diameter: float = 0.0         # um
    spp_charge: int = 1
    spp_diameter: float = 0.0            # um

    # ── Output ─────────────────────────────────────────────────────────────
    gds_filepath: str = ""
    pixel_size_um: float = 0.0

    # ── Utilities ──────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.__init__()

    @property
    def pixel_size(self) -> float:
        """Effective pixel pitch in microns."""
        return self.unitcells_per_pixel * (self.period / 1000.0)

    @property
    def mask_shape(self) -> Optional[tuple[int, int]]:
        return None if self.phase_mask is None else self.phase_mask.shape