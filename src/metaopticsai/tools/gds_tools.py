"""GDS export tool."""
from __future__ import annotations

import numpy as np

from metaopticsai.configs.settings import settings
from metaopticsai.fabrication.gds import GDSEngine
from metaopticsai.phase import profiles, quantization
from metaopticsai.tools.base import BaseTool
from metaopticsai.tools.schemas.gds import GDSExportInput, GDSExportOutput


class ExportGDSTool(BaseTool[GDSExportInput, GDSExportOutput]):
    name = "export_gds"
    description = "Export a metalens design as a fabrication-ready GDSII file."
    Input = GDSExportInput
    Output = GDSExportOutput
    needs_store = True

    def _run(self, p: GDSExportInput) -> GDSExportOutput:
        design = self.store.get(p.design_handle, expected_kind="metalens_design")
        sweep = self.store.get(p.sweep_handle, expected_kind="rcwa_sweep").payload

        diameter_um = float(design.metadata.get("diameter_um", 50.0))
        focal_um = float(design.metadata.get("focal_length_um", 100.0))
        wl_nm = sweep["wavelength_nm"]

        mask = profiles.fresnel_zone_lens(
            wavelength_nm=wl_nm,
            focal_length_um=focal_um,
            diameter_um=diameter_um,
            pixel_size_um=p.pixel_size_um,
            circular=p.is_circular,
        )
        quantized, bins = quantization.quantize(mask, n_levels=p.n_levels)

        out_path = (settings.paths.output_dir / f"{p.output_name}.gds").resolve()
        if not str(out_path).startswith(str(settings.paths.output_dir)):
            raise ValueError("output_name escapes outputs directory")

        info = GDSEngine().generate(
            quantized_mask=quantized, bins=bins,
            dimensions=sweep["diameters_nm"],
            phases=[float(np.degrees(p)) for p in sweep["phases_rad"]],
            period=p.period_nm,
            pixel_size_um=p.pixel_size_um,
            unitcells_per_pixel=1,
            shape=p.shape,
            is_circular=p.is_circular,
            output_path=str(out_path),
        )
        handle = self.store.put(
            "gds_file", out_path,
            design_handle=p.design_handle,
            output_name=p.output_name,
        )
        return GDSExportOutput(
            handle=handle,
            path=str(out_path),
            size_kb=float(info["file_size_kb"]),
            elapsed_s=float(info["time_seconds"]),
        )
