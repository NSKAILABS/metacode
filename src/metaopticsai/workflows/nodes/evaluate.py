"""Evaluate node — runs PSF & MTF analysis on the optimized design.

Generates the canonical FZL phase mask from the parsed brief and runs PSF +
MTF analysis. Stores handles on the state for downstream reflection.
"""
from __future__ import annotations

import logging
from typing import Callable

from metaopticsai.tools.base import ToolRegistry
from metaopticsai.workflows.state import AutoMLState

log = logging.getLogger(__name__)


def make_evaluate_node(tools: ToolRegistry) -> Callable[[AutoMLState], dict]:
    mask_tool = tools.get("generate_phase_mask")
    psf_tool = tools.get("analyze_psf")
    mtf_tool = tools.get("analyze_mtf")

    def evaluate_node(state: AutoMLState) -> dict:
        parsed = state["parsed"]
        wl_nm = parsed["wavelength_nm"]
        try:
            mask = mask_tool.call_sync({
                "mask_type": "fzl",
                "wavelength_nm": wl_nm,
                "diameter_um": parsed["diameter_um"],
                "focal_length_um": parsed["focal_length_um"],
                "pixel_size_um": 0.35,
                "circular": True,
            })
            psf = psf_tool.call_sync({
                "mask_handle": mask["handle"],
                "wavelength_nm": wl_nm,
                "pixel_size_um": 0.35,
                "focal_length_um": parsed["focal_length_um"],
                "pad_factor": 4,
            })
            mtf = mtf_tool.call_sync({
                "psf_handle": psf["handle"],
                "pixel_size_um": 0.35,
                "wavelength_nm": wl_nm,
                "focal_length_um": parsed["focal_length_um"],
            })
        except Exception as e:
            log.exception("Evaluate failed")
            return {"errors": state.get("errors", []) + [f"evaluate: {e}"]}

        return {
            "psf_handle": psf["handle"],
            "mtf_handle": mtf["handle"],
        }

    return evaluate_node