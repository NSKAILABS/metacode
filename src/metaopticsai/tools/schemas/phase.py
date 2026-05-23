"""Schemas for phase-mask generation."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PhaseMaskInput(BaseModel):
    """Strict schema — every field explicit and described so the MCP
    Inspector renders the full UI."""

    mask_type: Literal["fzl", "axicon", "spp"] = Field(
        "fzl",
        description="Profile type. fzl = focusing lens; axicon = Bessel-beam; spp = vortex.",
    )
    wavelength_nm: float = Field(532.0, gt=200, lt=20000,
        description="Design wavelength in nm.")
    diameter_um: float = Field(100.0, gt=1, lt=100000,
        description="Aperture diameter in µm.")
    pixel_size_um: float = Field(0.35, gt=0.05, lt=10,
        description="Spatial pitch in µm. Match to RCWA `period_nm`/1000.")
    focal_length_um: float = Field(110.0, gt=0,
        description="Focal length in µm. Only used for mask_type='fzl'.")
    cone_angle_deg: float = Field(2.0, gt=0, lt=45,
        description="Axicon cone half-angle. Only used for mask_type='axicon'.")
    spp_charge: int = Field(1, ge=-10, le=10,
        description="Spiral phase plate topological charge. Only used for mask_type='spp'.")
    circular: bool = Field(True, description="Apply circular aperture stop.")


class PhaseMaskOutput(BaseModel):
    handle: str
    shape: list[int]
    phase_min_rad: float
    phase_max_rad: float
    mask_type: str
