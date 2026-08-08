"""End-to-end demo of the LLM + MCP meta-optics agent.
Run modes:
  python examples/demo.py         
  python examples/demo.py --no-llm       
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from mcp_server.tools import dispatch, list_available_backends 
from llm.groq_client import load_env  # noqa: E402


DEFAULT_PROMPT = (
    "Design a TiO2 metalens for green light (532 nm). Start from a "
    "verified template, validate it, run a 20-diameter RCWA sweep, "
    "and report phase coverage and mean transmission."
)


def dry_run() -> int:
    """Pipeline exercise without the LLM. Useful when GROQ_API_KEY is unset."""
    print("=== dry-run (no LLM) ===")
    print(f"backends available: {list_available_backends()}\n")

    # 1. list templates
    r = dispatch("list_templates", {})
    print("[list_templates]")
    for t in r["templates"]:
        print(f"  - {t['name']:32s} {t['material']:5s} @ {t['wavelength_nm']:.0f} nm")

    # 2. get one
    r = dispatch("get_template", {"name": "tio2_metalens_532nm"})
    params = r["template"]["params"]
    print(f"\n[get_template] wavelength_nm={params['wavelength_nm']}, "
          f"material={params['pillar_material']}, D={params['diameter_mm']} mm")

    # 3. validate
    r = dispatch("validate_design", {"params": params})
    print(f"\n[validate_design] valid={r['valid']}, "
          f"NA={r['derived']['numerical_aperture']:.3f}, "
          f"aspect={r['derived']['aspect_ratio']:.2f}")
    for w in r["warnings"]:
        print(f"  warn: {w}")

    # 4. simulate
    r = dispatch("run_simulation", {"params": params, "n_samples": 20})
    if not r["ok"]:
        print(f"[run_simulation] FAILED: {r['error']}")
        return 1
    res = r["result"]
    print(f"\n[run_simulation] backend={res['backend']}, "
          f"n_samples={res['n_samples']}, "
          f"phase_coverage={res['phase_coverage']:.2%}, "
          f"mean_transmission={res['mean_transmission']:.2%}, "
          f"elapsed={res['elapsed_s']:.2f} s")

    # 5. tips
    r_tips = dispatch("get_optimization_tips", {"result": res})
    print("\n[get_optimization_tips]")
    for tip in r_tips["tips"]:
        print(f"  - {tip}")

    print("\n=== dry-run complete ===")
    return 0


def live_run(prompt: str, show_tools: bool = True) -> int:
    from llm.agent import run_agent
    try:
        result = run_agent(prompt)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if show_tools:
        print("\n--- tool calls ---")
        for i, tc in enumerate(result.tool_calls, 1):
            print(f"\n[{i}] {tc['name']}  (iter {tc['iteration']})")
            print(f"    args: {json.dumps(tc['args'])[:220]}")
            r = tc["result"]
            if isinstance(r, dict) and r.get("ok") is False:
                print(f"    ERROR: {r.get('error')}")
            elif tc["name"] == "run_simulation" and r.get("ok"):
                s = r["result"]
                print(f"    result: backend={s['backend']}, "
                      f"cov={s['phase_coverage']:.2%}, "
                      f"trans={s['mean_transmission']:.2%}")
            else:
                short = json.dumps(r, default=str)
                print(f"    result: {short[:220]}"
                      + ("..." if len(short) > 220 else ""))

    print("\n=== Agent response ===")
    print(result.final_text)
    print(f"\n(iterations={result.iterations}, "
          f"stopped_on={result.stopped_on}, "
          f"tool_calls={len(result.tool_calls)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip the LLM; exercise the tools directly.")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                    help="Prompt for the agent (live mode only).")
    ap.add_argument("--quiet", action="store_true",
                    help="Do not print per-tool trace in live mode.")
    args = ap.parse_args()

    load_env()

    if args.no_llm:
        return dry_run()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set — falling back to --no-llm dry run.")
        print("(Set it in config/.env or the repo-root .env to run live.)\n")
        return dry_run()

    return live_run(args.prompt, show_tools=not args.quiet)


if __name__ == "__main__":
    sys.exit(main())
