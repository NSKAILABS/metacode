"""Reflect node — Self-RAG critique. LLM decides accept / re_optimize /
re_sweep / re_retrieve."""
from __future__ import annotations

import json
import logging
from typing import Callable

from metaopticsai.llm.base import LLMProvider
from metaopticsai.prompts import REFLECT_PROMPT
from metaopticsai.utils.json_io import safe_parse_llm_json
from metaopticsai.workflows.state import AutoMLState

log = logging.getLogger(__name__)


def make_reflect_node(provider: LLMProvider) -> Callable[[AutoMLState], dict]:
    chain = REFLECT_PROMPT | provider.chat_model()

    def reflect_node(state: AutoMLState) -> dict:
        achieved = state.get("achieved_strehl", 0.0)
        target = state.get("target_strehl", 0.85)
        metrics = state.get("sweep_metrics", {})
        try:
            resp = chain.invoke({
                "state": json.dumps({
                    "parsed": state.get("parsed", {}),
                    "design_params": state.get("design_params", {}),
                    "achieved_strehl": achieved,
                }, indent=2),
                "strehl": achieved,
                "target": target,
                "coverage": metrics.get("phase_coverage_2pi_fraction", 0.0),
                "transmission": metrics.get("mean_transmission", 0.0),
            })
            text = getattr(resp, "content", str(resp))
            parsed = safe_parse_llm_json(text, default={"decision": "accept"})
        except Exception as e:
            log.warning("Reflect failed: %s — accepting current design", e)
            parsed = {"decision": "accept", "rationale": f"reflect_error: {e}"}

        decision = parsed.get("decision", "accept")
        if decision not in ("accept", "re_optimize", "re_sweep", "re_retrieve"):
            decision = "accept"

        # Optionally apply suggested parameter changes (only for re_sweep)
        new_design_params = dict(state.get("design_params", {}))
        if decision == "re_sweep":
            for k, v in (parsed.get("suggested_changes") or {}).items():
                if v is not None and k in new_design_params:
                    new_design_params[k] = v

        history = list(state.get("decision_history", []))
        history.append({"decision": decision, "rationale": parsed.get("rationale", "")})

        return {
            "decision": decision,
            "design_params": new_design_params,
            "decision_history": history,
        }

    return reflect_node