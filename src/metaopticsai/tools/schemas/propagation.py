"""Schemas for field propagation tools."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PropagateInput(BaseModel):
    mask_handle: str = Field(..., description="Handle from generate_phase_mask.")
    wavelength_nm: float = Field(..., gt=200, lt=20000)
    pixel_size_um: float = Field(..., gt=0.05, lt=10)
    propagation_distance_um: float = Field(..., gt=0,
        description="Distance to propagate beyond the metasurface, in µm.")
    refractive_index: float = Field(1.0, gt=0.5, le=4.0,
        description="Index of the propagation medium (1.0 = air, 1.46 = SiO2).")
