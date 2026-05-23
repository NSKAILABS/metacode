"""Schemas for analysis tools."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PSFInput(BaseModel):
    mask_handle: str = Field(..., description="Handle from generate_phase_mask.")
    wavelength_nm: float = Field(..., gt=200, lt=20000)
    pixel_size_um: float = Field(..., gt=0.05, lt=10)
    focal_length_um: float = Field(0.0, ge=0.0,
        description="Optional — used for absolute coordinate scaling.")
    pad_factor: int = Field(4, ge=1, le=8)


class MTFInput(BaseModel):
    psf_handle: str = Field(..., description="Handle from analyze_psf.")
    wavelength_nm: float = Field(..., gt=200, lt=20000)
    pixel_size_um: float = Field(..., gt=0.05, lt=10)
    focal_length_um: float = Field(0.0, ge=0.0)


class ZernikeInput(BaseModel):
    mask_handle: str
    n_terms: int = Field(15, ge=4, le=45,
        description="Number of Noll-indexed Zernike terms to fit.")


class FieldMetricInput(BaseModel):
    psf_handle: str = Field(..., description="Handle from analyze_psf or propagate_field.")