"""Material descriptors — backend-agnostic.

The actual CSV lookup happens inside `physics/materials.py` (which is the only
layer that knows about metabox3's CSV tables). Domain just holds the value.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class MaterialIndex(BaseModel):
    """Refractive index evaluation at a wavelength."""
    name: str
    wavelength_nm: float
    n: float = Field(..., description="Real part of refractive index")
    k: float = Field(0.0, description="Imaginary part (extinction)")
    valid_range_um: tuple[float, float]

    @property
    def complex_index(self) -> complex:
        return complex(self.n, self.k)

    def __str__(self) -> str:
        if self.k == 0:
            return f"{self.name} n={self.n:.4f} @ {self.wavelength_nm}nm"
        return f"{self.name} n={self.n:.4f}+{self.k:.4f}j @ {self.wavelength_nm}nm"