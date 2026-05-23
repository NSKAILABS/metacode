"""Schemas for material tools."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MaterialIndexInput(BaseModel):
    name: str = Field(..., description="Material CSV name from list_materials.")
    wavelength_nm: float = Field(..., gt=10, lt=50000)


class MaterialIndexOutput(BaseModel):
    name: str
    wavelength_nm: float
    n: float
    k: float
    complex: str
    valid_range_um: tuple[float, float]
