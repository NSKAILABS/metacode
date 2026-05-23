"""Simulate node — runs the RCWA sweep tool to build the phase library."""
from __future__ import annotations

import logging
from typing import Callable

from metaopticsai.tools.base import ToolRegistry
from metaopticsai.workflows.state import AutoMLState

log = logging.getLogger(__name__)


def make_simulate_node(tools: ToolRegistry) -> Callable[[AutoMLState], dict]:
    sweep_tool = tools.get("run_rcwa_sweep")

    def simulate_node(state: AutoMLState) -> dict:
        parsed = state["parsed"]
        params = state["design_params"]
        payload = {
            "wavelength_nm": parsed["wavelength_nm"],
            "pillar_material": params["pillar_material"],
            "pillar_height_nm": params["pillar_height_nm"],
            "period_nm": params["period_nm"],
            "min_diameter_nm": params["min_diameter_nm"],
            "max_diameter_nm": params["max_diameter_nm"],
            "n_samples": int(params["n_samples"]),
            "xy_harmonics": int(params["xy_harmonics"]),
        }
        try:
            result = sweep_tool.call_sync(payload)
        except Exception as e:
            log.exception("Simulation failed")
            return {"errors": state.get("errors", []) + [f"simulate: {e}"]}

        return {
            "sweep_handle": result["handle"],
            "sweep_metrics": {
                "phase_coverage_2pi_fraction": result["phase_coverage_2pi_fraction"],
                "mean_transmission": result["mean_transmission"],
                "backend": result["backend"],
                "elapsed_s": result["elapsed_s"],
            },
        }

    return simulate_node