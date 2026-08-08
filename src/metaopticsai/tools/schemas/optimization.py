"""Schemas for optimization & surrogate-training tools."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OptimizeInput(BaseModel):
    sweep_handle: str = Field(..., description="Handle from run_rcwa_sweep.")
    target_strehl: float = Field(0.85, ge=0.1, le=1.0)
    max_iters: int = Field(20, ge=1, le=200)
    learning_rate: float = Field(2e-9, gt=0, lt=1e-3)
    method: Literal["heuristic", "gradient", "surrogate"] = "heuristic"


class OptimizeOutput(BaseModel):
    handle: str
    achieved_strehl: float
    iterations_run: int
    target_met: bool
    method: str


class TrainMetamodelInput(BaseModel):
    wavelength_nm: float = Field(..., gt=200, lt=20000)
    pillar_material: Literal["TiO2", "Si", "GaN", "SiN"] = "TiO2"
    pillar_height_nm: float = Field(600.0, gt=50, lt=3000)
    period_nm: float = Field(350.0, gt=100, lt=2000)
    min_diameter_nm: float = Field(50.0, gt=10)
    max_diameter_nm: float = Field(160.0, gt=20)
    n_samples: int = Field(30, ge=10, le=100)
    n_epochs: int = Field(50, ge=10, le=500)


class JobSubmitOutput(BaseModel):
    job_id: str
    status: str
    extra: dict = Field(default_factory=dict)
