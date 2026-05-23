"""IncidenceSpec — backend-agnostic incidence parameters.

This is the unified replacement for `metabox3.utils.Incidence`, used at the
physics-backend boundary. SI units (meters, radians-via-degrees).
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class IncidenceSpec(BaseModel):
    """Incidence of illumination on a unit cell or surface."""

    wavelength_m: list[float] = Field(..., min_length=1)
    theta_deg: list[float] = Field(default_factory=lambda: [0.0])
    phi_deg: list[float] = Field(default_factory=lambda: [0.0])
    jones_vector: tuple[float, float] = (1.0, 0.0)

    @classmethod
    def at_wavelength_nm(cls, wavelength_nm: float) -> "IncidenceSpec":
        return cls(wavelength_m=[wavelength_nm * 1e-9])