"""
clients.cli — Headless CLI for MetaOpticsAI.

Three subcommands:
    design   — natural-language request → full hybrid pipeline (Claude + Ollama)
    sweep    — call run_rcwa_sweep directly, no LLM
    inspect  — list current store handles

Run:
    python clients/cli.py design "Design a 532nm TiO2 metalens, NA 0.4, ø 100µm, f 110µm"
    python clients/cli.py sweep --wavelength 532 --material TiO2 --height 600 --period 350
    python clients/cli.py inspect
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from controllers.hybrid_loop import run_design                          # noqa: E402

# For sweep / inspect we call MCP tools directly via fastmcp's Client.
from fastmcp import Client                                              # noqa: E402

SERVER_CMD = ["python", str(_PROJECT_ROOT / "mcp_server" / "server.py")]


async def cmd_design(args) -> None:
    summary = await run_design(args.requirement, args.max_iters)
    print("\n══════════ FINAL DESIGN SUMMARY ══════════\n")
    print(summary)


async def cmd_sweep(args) -> None:
    async with Client(SERVER_CMD) as mcp:
        out = await mcp.call_tool("run_rcwa_sweep", {"params": {
            "wavelength_nm":    args.wavelength,
            "pillar_material":  args.material,
            "pillar_height_nm": args.height,
            "period_nm":        args.period,
            "min_diameter_nm":  args.min_diameter,
            "max_diameter_nm":  args.max_diameter,
            "n_samples":        args.n_samples,
            "xy_harmonics":     args.harmonics,
        }})
        print(json.dumps(out.structured_content or out.data, indent=2))


async def cmd_inspect(args) -> None:
    async with Client(SERVER_CMD) as mcp:
        out = await mcp.call_tool("list_artifacts", {"kind": args.kind})
        print(json.dumps(out.structured_content or out.data, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="metaoptics", description="MetaOpticsAI CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_design = sub.add_parser("design", help="natural-language design")
    p_design.add_argument("requirement")
    p_design.add_argument("--max-iters", type=int, default=3)
    p_design.set_defaults(func=cmd_design)

    p_sweep = sub.add_parser("sweep", help="run an RCWA sweep")
    p_sweep.add_argument("--wavelength",   type=float, default=532)
    p_sweep.add_argument("--material",     default="TiO2", choices=["TiO2","Si","GaN","SiN"])
    p_sweep.add_argument("--height",       type=float, default=600.0)
    p_sweep.add_argument("--period",       type=float, default=350.0)
    p_sweep.add_argument("--min-diameter", type=float, default=50.0)
    p_sweep.add_argument("--max-diameter", type=float, default=160.0)
    p_sweep.add_argument("--n-samples",    type=int,   default=20)
    p_sweep.add_argument("--harmonics",    type=int,   default=3)
    p_sweep.set_defaults(func=cmd_sweep)

    p_inspect = sub.add_parser("inspect", help="list artefacts in the session store")
    p_inspect.add_argument("--kind", default=None)
    p_inspect.set_defaults(func=cmd_inspect)

    args = p.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
