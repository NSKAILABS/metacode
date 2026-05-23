"""JSON encoding helpers tolerant of numpy types."""
from __future__ import annotations

import json
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """Encode numpy scalars/arrays + tuples + Pydantic in one place."""

    def default(self, o: Any):  # noqa: ANN401
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if hasattr(o, "model_dump"):
            return o.model_dump()
        return super().default(o)


def dumps(obj: Any, *, indent: int | None = 2) -> str:
    return json.dumps(obj, cls=NumpyJSONEncoder, indent=indent)


def safe_parse_llm_json(text: str, fallback: dict) -> dict:
    """Extract a JSON object from an LLM response, tolerating ```json fences
    and leading/trailing prose. Returns `fallback` on any parse failure.
    """
    if not text:
        return fallback
    cleaned = text.strip()
    for fence in ("```json", "```"):
        cleaned = cleaned.replace(fence, "")
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        return fallback
    try:
        result = json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return fallback
    if not isinstance(result, dict):
        return fallback
    return result