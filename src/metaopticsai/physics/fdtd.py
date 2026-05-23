"""FDTD reference library — pre-computed phase-vs-dimension curves.

Used by:
  - the `get_fdtd_library` MCP tool, to expose the built-in library
  - notebooks doing quick lookup of known-good unit cells

Direct port of the old `core/meta_data.py`. Pure data + Excel ingest.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import openpyxl
from scipy.interpolate import interp1d


@dataclasses.dataclass
class FDTDEntry:
    """Single FDTD dataset for one wavelength."""
    shape: str
    material: str
    param_type: str
    height: float                              # nm
    period: float                              # nm
    dimensions: list[float]
    phases: list[float]                        # degrees
    transmissions: list[float]
    cross_width: Optional[float] = None
    fin_width: Optional[float] = None
    fin_length: Optional[float] = None
    reference: str = ""


BUILTIN_DATA: dict[int, FDTDEntry] = {
    405: FDTDEntry(
        shape="Fin", material="TiO2", param_type="a",
        height=600, period=200,
        dimensions=[0, 30, 60, 90, 120, 150, 180],
        phases=[0, 60, 120, 180, 240, 300, 360],
        transmissions=[1] * 7,
        fin_width=40, fin_length=150,
        reference="Ahmed et al., Opt. Mat. Express 10.2 (2020)",
    ),
    532: FDTDEntry(
        shape="Fin", material="TiO2", param_type="a",
        height=600, period=325,
        dimensions=[0, 30, 60, 90, 120, 150, 180],
        phases=[0, 60, 120, 180, 240, 300, 360],
        transmissions=[1] * 7,
        fin_width=95, fin_length=250,
        reference="Ahmed et al., Opt. Mat. Express 10.2 (2020)",
    ),
    633: FDTDEntry(
        shape="Cylinder", material="Silicon", param_type="a",
        height=280, period=250,
        dimensions=[40, 55, 60.8, 64.5, 67.5, 70.5, 75.3, 85.7],
        phases=[0, 45, 90, 135, 180, 225, 270, 315],
        transmissions=[1] * 8,
        reference="Standard Si nanodisk at 633nm",
    ),
    850: FDTDEntry(
        shape="Cylinder", material="Silicon", param_type="a",
        height=475, period=390,
        dimensions=[58, 67.5, 74, 77.5, 84, 87.5, 92, 98.5, 104.5, 112, 118, 126],
        phases=[0, 20, 47, 73, 143, 192, 235, 279, 301, 322, 337, 355],
        transmissions=[1] * 12,
        reference="Si cylinder at 850nm",
    ),
    1064: FDTDEntry(
        shape="Cylinder", material="Silicon", param_type="a",
        height=170, period=620,
        dimensions=[130, 140, 155, 165, 175, 183, 195, 214, 240],
        phases=[0, 9, 29, 56, 112, 177, 270, 324, 360],
        transmissions=[1] * 9,
        reference="Si cylinder at 1064nm",
    ),
}


class FDTDStore:
    """Manages the collection of FDTD datasets, mutable for user uploads."""

    def __init__(self):
        self.data: dict[int, FDTDEntry] = dict(BUILTIN_DATA)

    def wavelengths(self) -> list[int]:
        return sorted(self.data.keys())

    def get(self, wl: int) -> Optional[FDTDEntry]:
        return self.data.get(wl)

    def add_user_entry(self, wl: int, entry: FDTDEntry) -> None:
        self.data[wl] = entry

    def load_excel(
        self,
        filepath: str,
        wl: int,
        shape: str,
        material: str,
        height: float,
        period: float,
        *,
        cross_width: float = 0,
        fin_width: float = 0,
        fin_length: float = 0,
    ) -> FDTDEntry:
        """Load a phase-vs-dimension table from .xlsx or .xls."""
        dimensions: list[float] = []
        phases: list[float] = []

        if filepath.endswith(".xls"):
            import xlrd  # type: ignore
            sheet = xlrd.open_workbook(filepath).sheets()[0]
            for i in range(sheet.nrows):
                row = sheet.row_values(i)
                if isinstance(row[0], (int, float)) and isinstance(row[1], (int, float)):
                    dimensions.append(float(row[0]))
                    phases.append(float(row[1]))
        else:
            wb = openpyxl.load_workbook(filepath, read_only=True)
            ws = wb.active
            for row in ws.iter_rows(values_only=True):
                if row[0] is None or row[1] is None:
                    continue
                try:
                    dimensions.append(float(row[0]))
                    phases.append(float(row[1]))
                except (ValueError, TypeError):
                    continue
            wb.close()

        if not dimensions:
            raise ValueError("No valid (dimension, phase) data found.")

        entry = FDTDEntry(
            shape=shape, material=material, param_type="a",
            height=height, period=period,
            dimensions=dimensions, phases=phases,
            transmissions=[1.0] * len(dimensions),
            cross_width=cross_width if shape == "Cross" else None,
            fin_width=fin_width if shape == "Fin" else None,
            fin_length=fin_length if shape == "Fin" else None,
        )
        self.data[wl] = entry
        return entry


def interpolate_phase_to_dimension(
    dimensions: list[float], phases: list[float],
) -> tuple[np.ndarray, np.ndarray, interp1d]:
    """Build a cubic interp1d: phase → dimension."""
    ph = np.array(phases, dtype=float)
    dim = np.array(dimensions, dtype=float)
    sort_idx = np.argsort(ph)
    f_interp = interp1d(
        ph[sort_idx], dim[sort_idx], kind="cubic", fill_value="extrapolate",
    )
    ph_fine = np.linspace(ph.min(), ph.max(), 200)
    return f_interp(ph_fine), ph_fine, f_interp