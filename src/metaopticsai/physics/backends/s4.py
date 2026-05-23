"""S4Backend — stub for the planned phoebe-p/S4 cross-validation backend.

Implements `SimulationBackend.is_available()` to return False until the user
populates `_simulate()` with actual S4 calls. The architecture is in place;
the wiring is not.

When you're ready (per the project memory: AWS Free Tier, t3.micro, swap file,
`-j1` compile), implement the three methods using:

    import S4
    sim = S4.New(...)
    sim.AddMaterial(...)
    sim.AddLayer(...)
    sim.SetExcitationPlanewave(...)
    refl, trn = sim.GetPowerFlux("substrate")
"""
from __future__ import annotations

import numpy as np

from metaopticsai.domain.results import (
    FieldResult, RCWASweepResult, UnitCellResult,
)
from metaopticsai.physics.backends.base import SimulationBackend


class S4Backend(SimulationBackend):
    name = "s4"

    def is_available(self) -> bool:
        try:
            import S4  # noqa: F401
            return True
        except ImportError:
            return False

    def simulate_unit_cell(self, params, incidence) -> UnitCellResult:  # noqa: D401
        raise NotImplementedError(
            "S4 backend stub. Populate using `phoebe-p/S4` per the project "
            "memory's validation roadmap."
        )

    def sweep_unit_cells(self, params, incidence, sweep_param, sweep_values) -> RCWASweepResult:
        raise NotImplementedError("S4 backend stub.")

    def propagate_field(self, phase_mask, wavelength_m, pixel_size_m, distance_m,
                        refractive_index=1.0) -> FieldResult:
        raise NotImplementedError(
            "S4 is an RCWA solver; propagation should fall through to "
            "MetaboxBackend or AnalyticalBackend via BackendRegistry."
        )