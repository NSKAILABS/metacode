"""
GDSII layout generation engine using gdsfactory 9.34.1.
Converts a quantized phase mask into a fabrication-ready GDS file.
"""

import numpy as np
import gdsfactory as gf
from gdsfactory.component import Component
from scipy.interpolate import interp1d
from typing import Callable, Optional
import time
import os

# Activate generic PDK
try:
    gf.gpdk.PDK.activate()
except Exception:
    pass


class GDSEngine:
    """Generates GDSII layout files using gdsfactory."""

    def __init__(self):
        self._progress_callback: Optional[Callable[[float], None]] = None

    def set_progress_callback(self, callback: Callable[[float], None]):
        self._progress_callback = callback

    def _report_progress(self, pct: float):
        if self._progress_callback:
            self._progress_callback(min(pct, 100.0))

    @staticmethod
    def _cross_points(length_um: float, width_um: float) -> list:
        w, l = width_um / 2.0, length_um / 2.0
        return [
            (-w, -l), (w, -l), (w, -w), (l, -w),
            (l, w), (w, w), (w, l), (-w, l),
            (-w, w), (-l, w), (-l, -w), (-w, -w)
        ]

    @staticmethod
    def _fin_points(theta_deg: float, fin_length_um: float, fin_width_um: float) -> list:
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        fl, fw = fin_length_um / 2.0, fin_width_um / 2.0
        def rotate(x, y):
            return (x * cos_t + y * sin_t, -x * sin_t + y * cos_t)
        return [rotate(-fl, -fw), rotate(fl, -fw), rotate(fl, fw), rotate(-fl, fw)]

    def generate(self, quantized_mask, bins, dimensions, phases,
                 period, pixel_size_um, unitcells_per_pixel,
                 shape, is_circular, output_path,
                 cross_width=0, fin_width=0, fin_length=0):
        t_start = time.time()

        ph_arr = np.array(phases, dtype=float)
        dim_arr = np.array(dimensions, dtype=float)
        sort_idx = np.argsort(ph_arr)
        f_interp = interp1d(ph_arr[sort_idx], dim_arr[sort_idx],
                            kind='cubic', fill_value='extrapolate')

        rows, cols = quantized_mask.shape
        m = min(rows, cols)
        top = gf.Component("METALENS")
        cell_cache = {}
        ph_min, ph_max = float(np.min(ph_arr)), float(np.max(ph_arr))
        period_um = period / 1000.0

        for i in range(rows):
            self._report_progress((i / rows) * 100)
            for j in range(cols):
                if is_circular:
                    ci, cj = i - m / 2, j - m / 2
                    if ci**2 + cj**2 >= (m / 2)**2:
                        continue

                level = quantized_mask[rows - 1 - i, j]
                phase_val = bins[level - 1] if level <= len(bins) else bins[-1]
                phase_val = np.clip(phase_val, ph_min, ph_max)
                dim_nm = round(float(f_interp(phase_val)), 1)
                dim_um = dim_nm / 1000.0

                if dim_nm not in cell_cache:
                    unit = gf.Component(f"u_{dim_nm}")
                    if shape == 'Cylinder' and dim_um > 0.001:
                        unit.add_ref(gf.components.circle(radius=dim_um, layer=(1, 0)))
                    elif shape == 'Cross':
                        pts = self._cross_points(dim_um, cross_width / 1000.0)
                        if len(pts) > 2:
                            unit.add_polygon(pts, layer=(1, 0))
                    elif shape == 'Fin':
                        pts = self._fin_points(dim_nm, fin_length / 1000.0, fin_width / 1000.0)
                        if len(pts) > 2:
                            unit.add_polygon(pts, layer=(1, 0))

                    arr = gf.Component(f"a_{dim_nm}")
                    for ai in range(unitcells_per_pixel):
                        for aj in range(unitcells_per_pixel):
                            ref = arr.add_ref(unit)
                            ref.dmove((ai * period_um, aj * period_um))
                    cell_cache[dim_nm] = arr

                ref = top.add_ref(cell_cache[dim_nm])
                ref.dmove((j * pixel_size_um, i * pixel_size_um))

        self._report_progress(100)
        if not output_path.endswith('.gds'):
            output_path += '.gds'
        top.write_gds(output_path)

        elapsed = time.time() - t_start
        file_size_kb = os.path.getsize(output_path) / 1024 if os.path.exists(output_path) else 0

        return {'time_seconds': elapsed, 'file_size_kb': file_size_kb, 'filepath': output_path}