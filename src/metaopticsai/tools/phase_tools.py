"""Phase-mask generation tool."""
from __future__ import annotations

import json

from metaopticsai.phase import profiles
from metaopticsai.tools.base import BaseTool
from metaopticsai.tools.schemas.phase import PhaseMaskInput, PhaseMaskOutput


class GeneratePhaseMaskTool(BaseTool[PhaseMaskInput, PhaseMaskOutput]):
    name = "generate_phase_mask"
    description = (
        "Generate a 2-D continuous phase mask for one of three canonical "
        "elements: fzl (Fresnel zone lens), axicon (Bessel beam), or spp "
        "(spiral phase plate). Returns a handle to the phase array (radians). "
        "Stage 3 of the canonical workflow."
    )
    Input = PhaseMaskInput
    Output = PhaseMaskOutput
    needs_store = True

    def _run(self, p: PhaseMaskInput) -> PhaseMaskOutput | list:
        if p.mask_type == "fzl":
            mask = profiles.fresnel_zone_lens(
                wavelength_nm=p.wavelength_nm,
                focal_length_um=p.focal_length_um,
                diameter_um=p.diameter_um,
                pixel_size_um=p.pixel_size_um,
                circular=p.circular,
            )
        elif p.mask_type == "axicon":
            mask = profiles.axicon(
                wavelength_nm=p.wavelength_nm,
                cone_angle_deg=p.cone_angle_deg,
                diameter_um=p.diameter_um,
                pixel_size_um=p.pixel_size_um,
                circular=p.circular,
            )
        elif p.mask_type == "spp":
            mask = profiles.spiral_phase_plate(
                charge=p.spp_charge,
                diameter_um=p.diameter_um,
                pixel_size_um=p.pixel_size_um,
                circular=p.circular,
            )
        else:
            raise ValueError(f"Unknown mask_type {p.mask_type!r}")

        handle = self.store.put(
            "phase_mask", mask,
            mask_type=p.mask_type,
            wavelength_nm=p.wavelength_nm,
            diameter_um=p.diameter_um,
            pixel_size_um=p.pixel_size_um,
        )
        return PhaseMaskOutput(
            handle=handle,
            shape=list(mask.shape),
            phase_min_rad=float(mask.min()),
            phase_max_rad=float(mask.max()),
            mask_type=p.mask_type,
        )
