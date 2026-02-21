"""
Design state container — holds all parameters for the current metalens design.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class DesignState:
    """Mutable state object tracking the entire design pipeline."""

    # ── Wavelength & Material ──
    wavelength: int = 0            # nm
    shape: str = 'Cylinder'        # 'Cylinder', 'Cross', 'Fin'
    material: str = ''
    height: float = 0.0            # nm
    period: float = 0.0            # nm

    # ── FDTD data ──
    dimensions: Optional[List[float]] = None
    phases: Optional[List[float]] = None
    transmissions: Optional[List[float]] = None
    is_user_data: bool = False

    # ── Cross / Fin params ──
    cross_width: float = 0.0
    fin_width: float = 0.0
    fin_length: float = 0.0

    # ── Phase mask design ──
    phase_mask: Optional[np.ndarray] = None   # 2D array of continuous phase (radians)
    quantized_mask: Optional[np.ndarray] = None  # quantized integer labels
    bins: Optional[List[float]] = None

    # ── Design parameters ──
    unitcells_per_pixel: int = 1
    phase_levels: int = 8
    phase_min: float = 0.0         # degrees
    phase_max: float = 360.0       # degrees
    is_circular: bool = True

    # ── Library element params ──
    fzl_focal_length: float = 0.0  # um
    fzl_diameter: float = 0.0      # um
    axicon_angle: float = 0.0      # degrees
    axicon_diameter: float = 0.0   # um
    spp_charge: int = 1
    spp_diameter: float = 0.0      # um

    # ── Output ──
    gds_filepath: str = ''
    pixel_size_um: float = 0.0     # unitcells_per_pixel * period / 1000

    def reset(self):
        """Reset to default state."""
        self.__init__()

    @property
    def pixel_size(self) -> float:
        """Pixel pitch in um."""
        return self.unitcells_per_pixel * (self.period / 1000.0)

    @property
    def mask_shape(self) -> Optional[Tuple[int, int]]:
        if self.phase_mask is not None:
            return self.phase_mask.shape
        return None