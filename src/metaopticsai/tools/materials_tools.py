"""Material lookup tools."""
from __future__ import annotations

from pydantic import BaseModel

from metaopticsai.physics.materials import get_material_index, list_materials
from metaopticsai.tools.base import BaseTool
from metaopticsai.tools.schemas.materials import (
    MaterialIndexInput, MaterialIndexOutput,
)


class _Empty(BaseModel):
    pass


class ListMaterialsTool(BaseTool):
    name = "list_materials"
    description = (
        "List every refractive-index CSV bundled in metabox3/material_data. "
        "Use BEFORE picking a material so you know which names are valid."
    )
    Input = _Empty

    def _run(self, _params):
        names = list_materials()
        return {"materials": names, "count": len(names)}


class GetMaterialIndexTool(BaseTool[MaterialIndexInput, MaterialIndexOutput]):
    name = "get_material_index"
    description = (
        "Look up complex refractive index n + ik of a bundled material at a "
        "given wavelength via cubic CSV interpolation."
    )
    Input = MaterialIndexInput
    Output = MaterialIndexOutput

    def _run(self, p: MaterialIndexInput) -> MaterialIndexOutput:
        idx = get_material_index(p.name, p.wavelength_nm)
        return MaterialIndexOutput(
            name=idx.name,
            wavelength_nm=idx.wavelength_nm,
            n=idx.n, k=idx.k,
            complex=str(idx.complex_index),
            valid_range_um=idx.valid_range_um,
        )
