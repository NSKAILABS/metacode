"""Generate node — LLM picks the meta-atom sweep parameters."""
from __future__ import annotations

import logging
from typing import Callable

from metaopticsai.llm.base import LLMProvider
from metaopticsai.prompts import GENERATE_PARAMS_PROMPT
from metaopticsai.utils.json_io import safe_parse_llm_json
from metaopticsai.workflows.state import AutoMLState

log = logging.getLogger(__name__)


def make_generate_node(provider: LLMProvider) -> Callable[[AutoMLState], dict]:
    chain = GENERATE_PARAMS_PROMPT | provider.chat_model()

    def generate_node(state: AutoMLState) -> dict:
        parsed = state.get("parsed", {})
        docs = state.get("retrieved_docs", [])
        context = "\n\n---\n\n".join(
            f"[{d['source']}]\n{d['content']}" for d in docs
        ) or "(no retrieved context)"

        try:
            resp = chain.invoke({"parsed": parsed, "context": context})
            text = getattr(resp, "content", str(resp))
            params = safe_parse_llm_json(text, default={})
        except Exception as e:
            log.warning("Generate failed: %s — using defaults", e)
            params = {}

        defaults = {
            "pillar_material": parsed.get("preferred_material", "TiO2"),
            "pillar_height_nm": 600.0,
            "period_nm": 350.0,
            "min_diameter_nm": 50.0,
            "max_diameter_nm": 160.0,
            "n_samples": 30,
            "xy_harmonics": 5,
            "reasoning": "defaults",
        }
        merged = {**defaults, **params}

        # Sanity clamp: max_diameter < period - 30
        if merged["max_diameter_nm"] > merged["period_nm"] - 30:
            merged["max_diameter_nm"] = merged["period_nm"] - 30
            merged["reasoning"] = (
                merged.get("reasoning", "") + " [clamped max_diameter < period-30]"
            )

        return {"design_params": merged}

    return generate_node