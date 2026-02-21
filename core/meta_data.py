"""
FDTD data store and interpolation utilities.

Modernized from the original metaData.py. Stores phase-vs-dimension
curves for various wavelengths and nanostructure geometries.
"""

import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import openpyxl
import os


@dataclass
class FDTDEntry:
    """Single FDTD dataset for a specific wavelength."""
    shape: str              # 'Cylinder', 'Cross', 'Fin'
    material: str           # 'Silicon', 'TiO2', 'Nb2O5', etc.
    param_type: str         # 'a' = vary dimension at fixed period; 'b' = vary period at fixed dim
    height: float           # nm
    period: float           # nm (or fixed dim for type 'b')
    dimensions: List[float] # varying parameter values (radius/length/angle)
    phases: List[float]     # corresponding transmission phases (degrees)
    transmissions: List[float]  # transmission amplitudes
    cross_width: Optional[float] = None   # for Cross shape
    fin_width: Optional[float] = None     # for Fin shape
    fin_length: Optional[float] = None    # for Fin shape
    reference: str = ""


# ─── Built-in FDTD Library ───────────────────────────────────────────
BUILTIN_DATA = {
    405: FDTDEntry(
        shape='Fin', material='TiO2', param_type='a',
        height=600, period=200,
        dimensions=[0, 30, 60, 90, 120, 150, 180],
        phases=[0, 60, 120, 180, 240, 300, 360],
        transmissions=[1]*7,
        fin_width=40, fin_length=150,
        reference="Ahmed et al., Opt. Mat. Express 10.2 (2020)"
    ),
    532: FDTDEntry(
        shape='Fin', material='TiO2', param_type='a',
        height=600, period=325,
        dimensions=[0, 30, 60, 90, 120, 150, 180],
        phases=[0, 60, 120, 180, 240, 300, 360],
        transmissions=[1]*7,
        fin_width=95, fin_length=250,
        reference="Ahmed et al., Opt. Mat. Express 10.2 (2020)"
    ),
    633: FDTDEntry(
        shape='Cylinder', material='Silicon', param_type='a',
        height=280, period=250,
        dimensions=[40, 55, 60.8, 64.5, 67.5, 70.5, 75.3, 85.7],
        phases=[0, 45, 90, 135, 180, 225, 270, 315],
        transmissions=[1]*8,
        reference="Standard Si nanodisk at 633nm"
    ),
    715: FDTDEntry(
        shape='Cylinder', material='Silicon', param_type='a',
        height=130, period=360,
        dimensions=[115, 120, 125, 130, 135, 140, 145, 150, 155],
        phases=[0, 23, 51, 116, 238, 280, 303, 327, 336],
        transmissions=[1]*9,
        reference="Si cylinder at 715nm"
    ),
    850: FDTDEntry(
        shape='Cylinder', material='Silicon', param_type='a',
        height=475, period=390,
        dimensions=[58, 67.5, 74, 77.5, 84, 87.5, 92, 98.5, 104.5, 112, 118, 126],
        phases=[0, 20, 47, 73, 143, 192, 235, 279, 301, 322, 337, 355],
        transmissions=[1]*12,
        reference="Si cylinder at 850nm"
    ),
    1064: FDTDEntry(
        shape='Cylinder', material='Silicon', param_type='a',
        height=170, period=620,
        dimensions=[130, 140, 155, 165, 175, 183, 195, 214, 240],
        phases=[0, 9, 29, 56, 112, 177, 270, 324, 360],
        transmissions=[1]*9,
        reference="Si cylinder at 1064nm"
    ),
    8800: FDTDEntry(
        shape='Cross', material='Silicon', param_type='a',
        height=1850, period=7250,
        dimensions=[1000, 1523, 2045, 2568, 3091, 3614, 4136, 4659, 5182, 5705, 6227, 6750],
        phases=[0, 3, 10, 23, 53, 127, 225, 277, 302, 315, 325, 337],
        transmissions=[1]*12,
        cross_width=1000,
        reference="Si cross at 8.8um"
    ),
    410000: FDTDEntry(
        shape='Cross', material='Silicon', param_type='a',
        height=150000, period=300000,
        dimensions=[187000, 201000, 208000, 212000, 217000, 225000, 237000, 267000],
        phases=[0, 40, 88, 133, 181, 226, 269, 315],
        transmissions=[1]*8,
        cross_width=70000,
        reference="Si cross at 410um (THz)"
    ),
}


class MetaDataStore:
    """Manages the collection of FDTD datasets."""

    def __init__(self):
        self.data: dict[int, FDTDEntry] = dict(BUILTIN_DATA)

    def get_wavelengths(self) -> List[int]:
        return sorted(self.data.keys())

    def get_entry(self, wl: int) -> Optional[FDTDEntry]:
        return self.data.get(wl)

    def add_user_data(self, wl: int, entry: FDTDEntry):
        self.data[wl] = entry

    def load_from_excel(self, filepath: str, wl: int, shape: str, material: str,
                        height: float, period: float,
                        cross_width: float = 0, fin_width: float = 0,
                        fin_length: float = 0) -> FDTDEntry:
        """Load phase-vs-dimension data from an Excel file (.xlsx or .xls)."""
        dimensions = []
        phases = []

        if filepath.endswith('.xls'):
            import xlrd
            workbook = xlrd.open_workbook(filepath)
            sheet = workbook.sheets()[0]
            for i in range(sheet.nrows):
                row = sheet.row_values(i)
                if isinstance(row[0], (int, float)) and isinstance(row[1], (int, float)):
                    dimensions.append(float(row[0]))
                    phases.append(float(row[1]))
        else:
            wb = openpyxl.load_workbook(filepath, read_only=True)
            ws = wb.active
            for row in ws.iter_rows(values_only=True):
                if row[0] is not None and row[1] is not None:
                    try:
                        dimensions.append(float(row[0]))
                        phases.append(float(row[1]))
                    except (ValueError, TypeError):
                        continue
            wb.close()

        if not dimensions:
            raise ValueError("No valid data found in the Excel file.")

        entry = FDTDEntry(
            shape=shape, material=material, param_type='a',
            height=height, period=period,
            dimensions=dimensions, phases=phases,
            transmissions=[1.0] * len(dimensions),
            cross_width=cross_width if shape == 'Cross' else None,
            fin_width=fin_width if shape == 'Fin' else None,
            fin_length=fin_length if shape == 'Fin' else None,
        )
        self.data[wl] = entry
        return entry


def interpolate_phase_data(dimensions: List[float], phases: List[float]
                           ) -> Tuple[np.ndarray, np.ndarray, interp1d]:
    """
    Create cubic interpolation: phase → dimension.

    Returns:
        (interp_dims, interp_phases, interpolator_fn)
    """
    ph = np.array(phases, dtype=float)
    dim = np.array(dimensions, dtype=float)

    # Sort by phase for monotonic interpolation
    sort_idx = np.argsort(ph)
    ph_sorted = ph[sort_idx]
    dim_sorted = dim[sort_idx]

    f_interp = interp1d(ph_sorted, dim_sorted, kind='cubic', fill_value='extrapolate')

    ph_fine = np.linspace(ph_sorted.min(), ph_sorted.max(), 200)
    dim_fine = f_interp(ph_fine)

    return dim_fine, ph_fine, f_interp