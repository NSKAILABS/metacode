"""MetalensDesignParams — the central design parameter object.

Migrated from the dataclass at the top of the old `core/automl.py`. Pure data:
no validation against backends, no I/O. Pydantic v2 model for easy serialization
in tool inputs/outputs and LangGraph state.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator


PillarMaterial = Literal["TiO2", "Si", "GaN", "SiN"]


class MetalensDesignParams(BaseModel):
    """All physical parameters required to describe a metalens.

    Lengths are in their natural units (nm or mm). Conversion to SI happens at
    the physics-backend boundary, never at the domain level.
    """

    # ── Wavelength / optical axis ──
    wavelength_nm: float = Field(532.0, gt=200, lt=20000)
    focal_length_mm: float = Field(1.0, gt=0)
    diameter_mm: float = Field(0.5, gt=0)

    # ── Meta-atom geometry ──
    pillar_material: PillarMaterial = "TiO2"
    pillar_height_nm: float = Field(500.0, gt=50, lt=3000)
    min_diameter_nm: float = Field(80.0, gt=10, lt=2000)
    max_diameter_nm: float = Field(300.0, gt=20, lt=2000)
    periodicity_nm: float = Field(350.0, gt=100, lt=2000)

    # ── RCWA & sampling ──
    xy_harmonics: tuple[int, int] = (3, 3)
    n_radial_pixels: int = Field(50, ge=5, le=400)
    substrate_index: float = Field(1.46, gt=1.0)

    # ── Polarization ──
    jones_vector: tuple[float, float] = (1.0, 0.0)

    # ── Cross-field validation ────────────────────────────────────────────

    @model_validator(mode="after")
    def _check_geometry(self) -> "MetalensDesignParams":
        if self.min_diameter_nm >= self.max_diameter_nm:
            raise ValueError(
                f"min_diameter_nm ({self.min_diameter_nm}) must be strictly less than "
                f"max_diameter_nm ({self.max_diameter_nm})"
            )
        # Pillars must fit inside the unit cell with at least 20 nm wall margin
        if self.max_diameter_nm >= self.periodicity_nm - 20:
            raise ValueError(
                f"max_diameter_nm ({self.max_diameter_nm}) must be ≥ 20 nm "
                f"below periodicity_nm ({self.periodicity_nm}). The pillar "
                f"cannot fill the unit cell."
            )
        return self

    # ── Derived properties ─────────────────────────────────────────────────

    @computed_field  # type: ignore[misc]
    @property
    def numerical_aperture(self) -> float:
        D = self.diameter_mm
        f = self.focal_length_mm
        return float(D / (2.0 * math.sqrt((D / 2) ** 2 + f ** 2)))

    @computed_field  # type: ignore[misc]
    @property
    def f_number(self) -> float:
        return float(self.focal_length_mm / self.diameter_mm)

    @computed_field  # type: ignore[misc]
    @property
    def mean_diameter_nm(self) -> float:
        return float((self.min_diameter_nm + self.max_diameter_nm) / 2.0)

    @computed_field  # type: ignore[misc]
    @property
    def aspect_ratio(self) -> float:
        return float(self.pillar_height_nm / max(self.mean_diameter_nm, 1.0))

    # ── Legacy-friendly converters ─────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "MetalensDesignParams":
        """Tolerant constructor; ignores extra keys, fills sensible defaults."""
        allowed = set(cls.model_fields.keys())
        clean = {k: v for k, v in d.items() if k in allowed}
        # Coerce list/tuple for xy_harmonics
        if "xy_harmonics" in clean and isinstance(clean["xy_harmonics"], list):
            clean["xy_harmonics"] = tuple(clean["xy_harmonics"])
        if "jones_vector" in clean and isinstance(clean["jones_vector"], list):
            clean["jones_vector"] = tuple(clean["jones_vector"])
        return cls.model_validate(clean)

    def to_dict(self) -> dict:
        d = self.model_dump()
        # JSON-friendly: convert tuples to lists
        d["xy_harmonics"] = list(self.xy_harmonics)
        d["jones_vector"] = list(self.jones_vector)
        return d