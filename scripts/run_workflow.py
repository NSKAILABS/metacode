"""Headless AutoML driver — run the full LangGraph workflow from a single
requirement string and print a JSON summary.

    python scripts/run_workflow.py \\
        --requirement "Design a 532-nm metalens, f=100µm, D=50µm, Strehl≥0.9"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaopticsai.orchestration.controller import MetaOpticsController  # noqa: E402
from metaopticsai.utils.json_io import NumpyJSONEncoder  # noqa: E402
from metaopticsai.utils.logging import setup_logging  # noqa: E402


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--requirement", required=True, help="Natural-language design brief.")
    p.add_argument("--target-strehl", type=float, default=0.85)
    p.add_argument("--provider", default=None, help="LLM provider name (ollama/groq/anthropic)")
    args = p.parse_args()

    controller = MetaOpticsController.build(provider_name=args.provider)

    result = asyncio.run(
        controller.design_metalens(args.requirement, target_strehl=args.target_strehl)
    )

    # Print a compact final-state JSON to stdout
    compact = {
        "parsed": result.get("parsed"),
        "design_params": result.get("design_params"),
        "sweep_handle": result.get("sweep_handle"),
        "sweep_metrics": result.get("sweep_metrics"),
        "design_handle": result.get("design_handle"),
        "achieved_strehl": result.get("achieved_strehl"),
        "psf_handle": result.get("psf_handle"),
        "mtf_handle": result.get("mtf_handle"),
        "gds_path": result.get("gds_path"),
        "summary": result.get("summary"),
        "decision_history": result.get("decision_history", []),
        "errors": result.get("errors", []),
    }
    print(json.dumps(compact, indent=2, cls=NumpyJSONEncoder))


if __name__ == "__main__":
    main()