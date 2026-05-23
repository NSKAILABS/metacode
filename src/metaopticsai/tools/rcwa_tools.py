"""RCWA tool implementations."""
from __future__ import annotations

import numpy as np

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.incidence import IncidenceSpec
from metaopticsai.physics.backends.base import BackendRegistry
from metaopticsai.tools.base import BaseTool
from metaopticsai.tools.schemas.rcwa import RCWASweepInput, RCWASweepOutput


class RunRCWASweepTool(BaseTool[RCWASweepInput, RCWASweepOutput]):
    name = "run_rcwa_sweep"
    description = (
        "Run a 1-wavelength RCWA sweep over pillar diameter to produce a "
        "phase-vs-diameter design library. Returns a handle plus phase "
        "coverage and mean transmission. Stage 2 of the canonical workflow."
    )
    Input = RCWASweepInput
    Output = RCWASweepOutput
    needs_store = True

    def __init__(self, store, backends: BackendRegistry):
        super().__init__(store=store)
        self.backends = backends

    def _run(self, p: RCWASweepInput) -> RCWASweepOutput:
        backend = self.backends.get(p.backend)
        params = MetalensDesignParams(
            wavelength_nm=p.wavelength_nm,
            pillar_material=p.pillar_material,
            pillar_height_nm=p.pillar_height_nm,
            periodicity_nm=p.period_nm,
            min_diameter_nm=p.min_diameter_nm,
            max_diameter_nm=p.max_diameter_nm,
            xy_harmonics=(p.xy_harmonics, p.xy_harmonics),
        )
        incidence = IncidenceSpec(wavelength_m=[p.wavelength_nm * 1e-9])
        sweep_vals = np.linspace(p.min_diameter_nm, p.max_diameter_nm, p.n_samples)

        result = backend.sweep_unit_cells(
            params, incidence,
            sweep_param="diameter", sweep_values=sweep_vals,
        )

        handle = self.store.put(
            "rcwa_sweep",
            {
                "diameters_nm": sweep_vals.tolist(),
                "amplitudes": result.amplitudes.tolist(),
                "phases_rad": result.phases_rad.tolist(),
                "wavelength_nm": p.wavelength_nm,
            },
            wavelength_nm=p.wavelength_nm,
            material=p.pillar_material,
            backend=result.backend,
        )

        advice = (
            "Phase coverage looks good for metalens design."
            if result.phase_coverage >= 0.9
            else "Phase coverage < 0.9 — increase pillar_height_nm or widen the diameter range."
        )
        return RCWASweepOutput(
            handle=handle,
            phase_coverage_2pi_fraction=result.phase_coverage,
            mean_transmission=result.mean_transmission,
            elapsed_s=round(result.elapsed_s, 2),
            n_samples=p.n_samples,
            backend=result.backend,
            advice=advice,
        )


class GetFDTDLibraryTool(BaseTool):
    """List the built-in FDTD reference library."""
    name = "get_fdtd_library"
    description = (
        "List the wavelengths, materials, and unit-cell shapes available in "
        "the built-in FDTD reference library."
    )

    # No input schema — empty Pydantic model
    from pydantic import BaseModel as _BM
    class _Empty(_BM):
        pass
    Input = _Empty

    def _run(self, _params):
        from metaopticsai.physics.fdtd import FDTDStore
        store = FDTDStore()
        out = []
        for wl in store.wavelengths():
            e = store.get(wl)
            if e is None:
                continue
            out.append({
                "wavelength_nm": wl,
                "shape": e.shape,
                "material": e.material,
                "height_nm": e.height,
                "period_nm": e.period,
                "n_samples": len(e.dimensions),
                "reference": e.reference,
            })
        return {"entries": out, "count": len(out)}
