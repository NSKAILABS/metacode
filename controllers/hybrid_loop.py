"""
controllers.hybrid_loop — Hybrid Claude + Ollama controller for MetaOpticsAI.

Pattern
-------
1. PLAN  (Claude — 1 call)
   Turn natural-language requirements into a DesignSpec + an ordered tool plan.
2. EXECUTE (MCP tools — many calls)
   Run get_fdtd_library → run_rcwa_sweep → generate_phase_mask → analyze_psf.
3. REFLECT (Ollama — many cheap calls)
   Given simulation results, propose parameter adjustments. Loop until target.
4. SUMMARY (Claude — 1 call)
   Generate a final human-readable design report.

This deliberately keeps Claude calls O(1) per design and Ollama calls O(N).

Usage
-----
    python controllers/hybrid_loop.py \\
        --requirement "Design a 532nm TiO2 metalens for NA 0.4..." \\
        --max-iters 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Ollama client — same setup core/automl.py already uses.
from langchain_ollama import ChatOllama

# We talk to the MCP server in-process via fastmcp's Client (stdio by default).
# This is the same surface a Claude Desktop chat would see.
from fastmcp import Client

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("hybrid")

CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class DesignSpec:
    wavelength_nm: float
    diameter_um: float
    focal_length_um: float
    target_strehl: float
    pillar_material: str = "TiO2"
    pillar_height_nm: float = 600.0
    period_nm: float = 350.0
    pixel_size_um: float = 0.35


@dataclass
class IterationResult:
    iteration: int
    spec: DesignSpec
    phase_coverage: float
    achieved_strehl: float
    reflection: str = ""


# ── Step 1 — PLAN (Claude) ────────────────────────────────────────────────

PLANNER_SYSTEM = """You are an expert metasurface engineer. You will be given a
natural-language metalens design requirement. Return ONLY a JSON object with
the following structure (no markdown, no commentary):

{
  "spec": {
    "wavelength_nm":    <float, design wavelength in nm>,
    "diameter_um":      <float, lens aperture diameter in micrometres>,
    "focal_length_um":  <float, focal length in micrometres>,
    "target_strehl":    <float in (0,1]>,
    "pillar_material":  <"TiO2"|"Si"|"GaN"|"SiN">,
    "pillar_height_nm": <float, pillar height>,
    "period_nm":        <float, unit-cell period>,
    "pixel_size_um":    <float, design grid pitch>
  },
  "rationale": "<one sentence explaining the choices>"
}

If the user did not specify a value, choose sensible defaults for the given
wavelength using these heuristics:
  • Visible (400-700 nm): TiO2, period 300-400 nm, height 400-600 nm.
  • Near-IR (900-1600 nm): Si, period 600-900 nm, height 500-800 nm.
  • Pixel size ≈ period.
"""


def plan_with_claude(client: anthropic.Anthropic, requirement: str) -> DesignSpec:
    log.info("[PLAN]  Calling Claude (%s) ...", CLAUDE_MODEL)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": requirement}],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text"))
    log.debug("Claude planning response: %s", text)
    # Extract the JSON object — be lenient about preamble
    a, b = text.find("{"), text.rfind("}") + 1
    obj = json.loads(text[a:b])
    spec = DesignSpec(**obj["spec"])
    log.info("[PLAN]  spec=%s  rationale=%s", asdict(spec), obj.get("rationale", ""))
    return spec


# ── Step 3 — REFLECT (Ollama, cheap) ──────────────────────────────────────

REFLECT_SYSTEM = """You are a metalens design critic. Given the current design
parameters and the latest simulation result, decide whether to:
 (a) accept the design (if achieved_strehl ≥ target_strehl), or
 (b) propose ONE concrete parameter change for the next iteration.

Respond ONLY with a JSON object:
{
  "accept":   <true|false>,
  "change":   {"<param_name>": <new_value>} | {},
  "reason":   "<one sentence>"
}

Valid param_name values: pillar_height_nm, period_nm, pillar_material,
pixel_size_um. Bounds: pillar_height_nm in [200, 1500],
period_nm in [200, 1500]."""


def reflect_with_ollama(llm: ChatOllama, spec: DesignSpec,
                         result: IterationResult) -> tuple[bool, dict]:
    user = (
        f"Current spec: {json.dumps(asdict(spec))}\n"
        f"Latest iteration: phase_coverage={result.phase_coverage:.3f}, "
        f"achieved_strehl={result.achieved_strehl:.3f}, "
        f"target={spec.target_strehl:.3f}\n"
    )
    raw = llm.invoke([
        {"role": "system", "content": REFLECT_SYSTEM},
        {"role": "user", "content": user},
    ])
    txt = raw.content if hasattr(raw, "content") else str(raw)
    a, b = txt.find("{"), txt.rfind("}") + 1
    try:
        obj = json.loads(txt[a:b])
    except Exception:                                                     # noqa: BLE001
        obj = {"accept": False, "change": {}, "reason": "parse_error"}
    return bool(obj.get("accept", False)), obj


# ── Step 4 — SUMMARY (Claude) ─────────────────────────────────────────────

SUMMARY_SYSTEM = """You are a photonics engineer writing a concise technical
design summary (3-5 sentences). Be precise about numerical values and
clearly state whether the target Strehl was achieved."""


def summarize_with_claude(client: anthropic.Anthropic, spec: DesignSpec,
                          history: list[IterationResult],
                          gds_path: str) -> str:
    payload = {
        "final_spec": asdict(spec),
        "iterations": [
            {"iter": h.iteration, "strehl": h.achieved_strehl,
             "coverage": h.phase_coverage} for h in history
        ],
        "gds_path": gds_path,
    }
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=500, system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    return "".join(b.text for b in msg.content if hasattr(b, "text"))


# ── Main loop ─────────────────────────────────────────────────────────────

async def run_design(requirement: str, max_iters: int) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment / .env")

    claude = anthropic.Anthropic()
    ollama = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                        temperature=0.1, num_predict=400)

    # 1. PLAN
    spec = plan_with_claude(claude, requirement)

    # 2 + 3. EXECUTE + REFLECT loop, all via the MCP server
    log.info("[EXEC] Connecting to MCP server (stdio)")
    server_path = _PROJECT_ROOT / "mcp_server" / "server.py"

    history: list[IterationResult] = []
    sweep_handle = design_handle = None

    async with Client(["python", str(server_path)]) as mcp:                # noqa: SIM117
        for it in range(1, max_iters + 1):
            log.info("[ITER %d] sweep with current spec", it)
            sweep = await mcp.call_tool("run_rcwa_sweep", {"params": {
                "wavelength_nm":    spec.wavelength_nm,
                "pillar_material":  spec.pillar_material,
                "pillar_height_nm": spec.pillar_height_nm,
                "period_nm":        spec.period_nm,
                "min_diameter_nm":  50.0,
                "max_diameter_nm":  min(spec.period_nm * 0.85, 200.0),
                "n_samples":        20, "xy_harmonics": 3,
            }})
            sweep = sweep.structured_content or sweep.data
            sweep_handle = sweep["handle"]
            log.info("[ITER %d] coverage=%.2f mean_amp=%.2f",
                     it, sweep["phase_coverage_2pi_fraction"], sweep["mean_transmission"])

            opt = await mcp.call_tool("optimize_metalens", {"params": {
                "sweep_handle": sweep_handle,
                "target_strehl": spec.target_strehl,
                "max_iters": 10,
            }})
            opt = opt.structured_content or opt.data
            design_handle = opt["handle"]

            result = IterationResult(
                iteration=it, spec=spec,
                phase_coverage=sweep["phase_coverage_2pi_fraction"],
                achieved_strehl=opt["achieved_strehl"],
            )
            accept, decision = reflect_with_ollama(ollama, spec, result)
            result.reflection = decision.get("reason", "")
            history.append(result)
            log.info("[ITER %d] strehl=%.3f  reflect: %s",
                     it, result.achieved_strehl, result.reflection)

            if accept:
                log.info("[ITER %d] accepted by reflector", it)
                break
            for k, v in decision.get("change", {}).items():
                if hasattr(spec, k):
                    log.info("  → updating %s: %s -> %s", k, getattr(spec, k), v)
                    setattr(spec, k, v)

        # Always export GDS for the best design found so far
        log.info("[EXEC] Exporting GDS")
        gds = await mcp.call_tool("export_gds", {"params": {
            "design_handle": design_handle, "sweep_handle": sweep_handle,
            "output_name": f"hybrid_{spec.wavelength_nm:.0f}nm",
            "period_nm": spec.period_nm, "pixel_size_um": spec.pixel_size_um,
            "n_levels": 8, "shape": "Cylinder", "is_circular": True,
        }})
        gds = gds.structured_content or gds.data
        gds_path = gds["path"]

    # 4. SUMMARY
    summary = summarize_with_claude(claude, spec, history, gds_path)
    log.info("[SUMMARY]\n%s", summary)
    return summary


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--requirement", required=True,
                   help="Natural-language design requirement.")
    p.add_argument("--max-iters", type=int, default=5)
    args = p.parse_args()
    summary = asyncio.run(run_design(args.requirement, args.max_iters))
    print("\n══════════ FINAL DESIGN SUMMARY ══════════\n")
    print(summary)


if __name__ == "__main__":
    cli()