"""Schemas for GDS export."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GDSExportInput(BaseModel):
    design_handle: str = Field(..., description="Handle from optimize_metalens.")
    sweep_handle: str = Field(..., description="Handle from run_rcwa_sweep.")
    output_name: str = Field("metalens", pattern=r"^[A-Za-z0-9_\-]{1,80}$")
    period_nm: float = Field(350.0, gt=100)
    pixel_size_um: float = Field(0.35, gt=0.05)
    n_levels: int = Field(8, ge=2, le=64)
    shape: Literal["Cylinder", "Cross", "Fin"] = "Cylinder"
    is_circular: bool = True


class GDSExportOutput(BaseModel):
    handle: str
    path: str
    size_kb: float
    elapsed_s: float
