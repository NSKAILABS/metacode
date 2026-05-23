"""Optimize node — drives Strehl toward the target."""
from __future__ import annotations

import logging
from typing import Callable

from metaopticsai.tools.base import ToolRegistry
from metaopticsai.workflows.state import AutoMLState

log = logging.getLogger(__name__)


def make_optimize_node(tools: ToolRegistry) -> Callable[[AutoMLState], dict]:
    opt_tool = tools.get("optimize_metalens")

    def optimize_node(state: AutoMLState) -> dict:
        sweep_handle = state.get("sweep_handle")
        if not sweep_handle:
            return {"errors": state.get("errors", []) + ["optimize: missing sweep_handle"]}

        try:
            result = opt_tool.call_sync({
                "sweep_handle": sweep_handle,
                "target_strehl": state.get("target_strehl", 0.85),
                "max_iters": 20,
                "method": "heuristic",
            })
        except Exception as e:
            log.exception("Optimize failed")
            return {"errors": state.get("errors", []) + [f"optimize: {e}"]}

        return {
            "design_handle": result["handle"],
            "achieved_strehl": float(result["achieved_strehl"]),
            "optimization_iter": state.get("optimization_iter", 0) + 1,
        }

    return optimize_node