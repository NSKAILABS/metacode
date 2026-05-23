"""Finalize node — exports GDS and writes a human-readable summary."""
from __future__ import annotations

import json
import logging
from typing import Callable

from metaopticsai.llm.base import LLMProvider
from metaopticsai.prompts import SUMMARIZE_DESIGN_PROMPT
from metaopticsai.tools.base import ToolRegistry
from metaopticsai.workflows.state import AutoMLState

log = logging.getLogger(__name__)


def make_finalize_node(
    tools: ToolRegistry, provider: LLMProvider,
) -> Callable[[AutoMLState], dict]:
    gds_tool = tools.get("export_gds")
    summary_chain = SUMMARIZE_DESIGN_PROMPT | provider.chat_model()

    def finalize_node(state: AutoMLState) -> dict:
        update: dict = {}

        # GDS export — best-effort
        try:
            design_handle = state.get("design_handle")
            sweep_handle = state.get("sweep_handle")
            if design_handle and sweep_handle:
                gds = gds_tool.call_sync({
                    "design_handle": design_handle,
                    "sweep_handle": sweep_handle,
                    "output_name": "automl_metalens",
                    "period_nm": state["design_params"]["period_nm"],
                    "pixel_size_um": 0.35,
                    "n_levels": 8,
                    "shape": "Cylinder",
                    "is_circular": True,
                })
                update["gds_handle"] = gds["handle"]
                update["gds_path"] = gds["path"]
        except Exception as e:
            log.warning("GDS export failed: %s", e)

        # Summary — also best-effort
        try:
            resp = summary_chain.invoke({
                "requirement": state.get("requirement", ""),
                "design": json.dumps(state.get("design_params", {}), indent=2),
                "strehl": state.get("achieved_strehl", 0.0),
                "transmission": state.get("sweep_metrics", {}).get("mean_transmission", 0.0),
                "coverage": state.get("sweep_metrics", {}).get("phase_coverage_2pi_fraction", 0.0),
                "gds_path": update.get("gds_path", "(no GDS exported)"),
            })
            update["summary"] = getattr(resp, "content", str(resp))
        except Exception as e:
            log.warning("Summary failed: %s", e)
            update["summary"] = (
                f"Design produced Strehl={state.get('achieved_strehl', 0.0):.2f}; "
                f"see handles in state."
            )

        return update

    return finalize_node