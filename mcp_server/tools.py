"""MCP-compatible tool definitions for the LLM agent.

Each tool is one Python function plus a JSON-Schema descriptor. The
schemas are the same shape as OpenAI/Groq tool-calling and MCP
`tool.list`, so the same registry drives:

  - the in-process agent (llm/agent.py) via `dispatch(name, args)`
  - an optional stdio MCP server (mcp_server/server.py) via the
    same TOOLS list.

Design principle: tools never raise past the boundary — a bad call
returns {"ok": false, "error": "..."} so the LLM can read the message
and retry.
"""
from __future__ import annotations

import traceback
from typing import Any

from pydantic import ValidationError

from metaopticsai.domain.design import MetalensDesignParams

from mcp_server.templates import list_templates, get_template
from mcp_server.workflow import get_workflow_guide
from mcp_server.simulator import run_diameter_sweep, available_backends


# ─── Tool implementations ──────────────────────────────────────────────

def _tool_list_templates(_args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "templates": list_templates()}


def _tool_get_template(args: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"ok": True, "template": get_template(args["name"])}
    except KeyError as e:
        return {"ok": False, "error": str(e)}


def _tool_get_workflow_guide(_args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "guide": get_workflow_guide()}


def _tool_validate_design(args: dict[str, Any]) -> dict[str, Any]:
    params = args.get("params", {})
    warnings: list[str] = []
    try:
        design = MetalensDesignParams(**params)
    except ValidationError as e:
        errs = [
            {"field": ".".join(str(p) for p in err["loc"]), "msg": err["msg"]}
            for err in e.errors()
        ]
        return {"ok": True, "valid": False, "errors": errs, "warnings": []}

    if design.aspect_ratio > 10:
        warnings.append(
            f"Aspect ratio {design.aspect_ratio:.1f} > 10 may be hard to fabricate."
        )
    if design.numerical_aperture > 0.85:
        warnings.append(
            f"NA={design.numerical_aperture:.2f} is very high; expect phase "
            "quantization to hurt Strehl."
        )
    return {
        "ok": True,
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "derived": {
            "numerical_aperture": design.numerical_aperture,
            "f_number": design.f_number,
            "aspect_ratio": design.aspect_ratio,
            "mean_diameter_nm": design.mean_diameter_nm,
        },
    }


def _tool_run_simulation(args: dict[str, Any]) -> dict[str, Any]:
    params = args.get("params", {})
    n = int(args.get("n_samples", 20))
    backend = args.get("backend", "auto")
    try:
        result = run_diameter_sweep(params, n_samples=n, backend=backend)
    except (ValidationError, KeyError, ValueError, RuntimeError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    except Exception as e:  # pragma: no cover — last-resort net
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=3),
        }
    # Trim huge arrays before returning to the LLM — full arrays remain
    # available if the caller re-runs; the LLM only needs the summary
    # metrics to decide the next step.
    summary = {k: v for k, v in result.items() if k not in ("sweep_values_nm", "amplitudes", "phases_rad")}
    summary["n_samples"] = result["n_samples"]
    summary["sweep_head"] = {
        "sweep_values_nm": result["sweep_values_nm"][:5],
        "phases_rad":     result["phases_rad"][:5],
        "amplitudes":     result["amplitudes"][:5],
    }
    return {"ok": True, "result": summary, "_full": result}


def _tool_get_optimization_tips(args: dict[str, Any]) -> dict[str, Any]:
    r = args.get("result", {})
    tips: list[str] = []
    cov = r.get("phase_coverage", 0.0)
    trans = r.get("mean_transmission", 0.0)

    if cov < 0.9:
        tips.append(
            f"Phase coverage {cov:.0%} < 90%: widen the diameter sweep range "
            "(reduce min_diameter_nm and/or increase max_diameter_nm), or "
            "increase pillar_height_nm to accumulate more propagation phase."
        )
    if trans < 0.7:
        tips.append(
            f"Mean transmission {trans:.0%} < 70%: pillars may be too tall "
            "or too close to the unit-cell wall. Reduce pillar_height_nm or "
            "cap max_diameter_nm to about (periodicity_nm - 40)."
        )
    if cov >= 0.95 and trans >= 0.85:
        tips.append(
            "Both metrics are strong; the design is ready to move to phase-"
            "mask synthesis and GDS export."
        )
    if not tips:
        tips.append(
            "Metrics are borderline. Try increasing n_samples to 30-40 for a "
            "smoother phase curve, then re-simulate."
        )
    return {"ok": True, "tips": tips}


# ─── Public registry ───────────────────────────────────────────────────

_HANDLERS = {
    "list_templates": _tool_list_templates,
    "get_template": _tool_get_template,
    "get_workflow_guide": _tool_get_workflow_guide,
    "validate_design": _tool_validate_design,
    "run_simulation": _tool_run_simulation,
    "get_optimization_tips": _tool_get_optimization_tips,
}


# JSON-Schema tool descriptors in the shape expected by both the Groq/OpenAI
# tool-calling API and the MCP protocol.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_templates",
            "description": (
                "List all verified starting-point meta-atom templates "
                "(name, wavelength, material, shape). Call this first when "
                "the user requests a new design."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_template",
            "description": "Return the full parameter dict for one template.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "Template name from list_templates."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_workflow_guide",
            "description": (
                "Return the canonical five-stage meta-optics design workflow. "
                "Consult this whenever unsure which tool to call next."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_design",
            "description": (
                "Validate a candidate design against MetalensDesignParams "
                "constraints. Always call this before run_simulation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "params": {
                        "type": "object",
                        "description": (
                            "Full MetalensDesignParams dict: wavelength_nm, "
                            "focal_length_mm, diameter_mm, pillar_material, "
                            "pillar_height_nm, min_diameter_nm, "
                            "max_diameter_nm, periodicity_nm."
                        ),
                    },
                },
                "required": ["params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_simulation",
            "description": (
                "Run an RCWA diameter sweep via the metabox3 backend (or the "
                "analytical fallback). Returns phase_coverage, "
                "mean_transmission and a summary of the sweep."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "params":    {"type": "object",
                                  "description": "Validated design params."},
                    "n_samples": {"type": "integer", "default": 20,
                                  "minimum": 5, "maximum": 60,
                                  "description": "Number of diameter samples."},
                    "backend":   {"type": "string", "default": "auto",
                                  "enum": ["auto", "metabox", "analytical"]},
                },
                "required": ["params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_optimization_tips",
            "description": (
                "Given a run_simulation result, return concrete suggestions "
                "for improving phase coverage or transmission."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "object",
                        "description": (
                            "The `result` object returned by run_simulation."
                        ),
                    },
                },
                "required": ["result"],
            },
        },
    },
]


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Route a tool call to its handler. Never raises."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"Unknown tool {name!r}. "
                f"Known: {list(_HANDLERS)}"}
    try:
        return handler(arguments or {})
    except Exception as e:  # pragma: no cover — belt & braces
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def list_available_backends() -> list[str]:
    """Convenience for the demo/README — not a tool."""
    try:
        return available_backends()
    except Exception:
        return ["analytical"]
