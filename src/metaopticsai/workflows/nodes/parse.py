"""Parse node — turns raw user text into a structured design brief."""
from __future__ import annotations

import logging
from typing import Callable

from metaopticsai.llm.base import LLMProvider
from metaopticsai.prompts import PARSE_REQUIREMENTS_PROMPT
from metaopticsai.utils.json_io import safe_parse_llm_json
from metaopticsai.workflows.state import AutoMLState

log = logging.getLogger(__name__)


def make_parse_node(provider: LLMProvider) -> Callable[[AutoMLState], dict]:
    chain = PARSE_REQUIREMENTS_PROMPT | provider.chat_model()

    def parse_node(state: AutoMLState) -> dict:
        requirement = state["requirement"]
        try:
            resp = chain.invoke({"requirement": requirement})
            text = getattr(resp, "content", str(resp))
            parsed = safe_parse_llm_json(text, default={})
        except Exception as e:
            log.warning("Parse failed: %s — falling back to defaults", e)
            parsed = {}

        # Defaults if any field is missing
        defaults = {
            "wavelength_nm": 532,
            "focal_length_um": 100.0,
            "diameter_um": 50.0,
            "target_strehl": state.get("target_strehl", 0.85),
            "preferred_material": "TiO2",
            "operating_band": "visible",
            "polarization": "unpolarized",
            "notes": "applied defaults for missing fields",
        }
        merged = {**defaults, **parsed}
        return {"parsed": merged}

    return parse_node