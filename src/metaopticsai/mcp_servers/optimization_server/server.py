"""Optimization-focused MCP server — analysis + optimization tools."""
from __future__ import annotations

import argparse

from fastmcp import FastMCP

from metaopticsai.configs.settings import settings
from metaopticsai.mcp_servers.common.subset import register_subset
from metaopticsai.orchestration.controller import MetaOpticsController
from metaopticsai.utils.logging import setup_logging

TOOLS = [
    "generate_phase_mask",
    "analyze_psf", "analyze_mtf", "zernike_decompose",
    "propagate_field",
    "compute_max_intensity", "compute_center_intensity", "compute_mtf_volume",
    "optimize_metalens", "train_metamodel",
    "submit_job", "get_job", "list_artifacts",
]


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=("stdio", "http", "sse"), default="stdio")
    p.add_argument("--host", default=settings.server.host)
    p.add_argument("--port", type=int, default=settings.server.port + 2)
    args = p.parse_args()

    controller = MetaOpticsController.build()
    mcp = FastMCP("MetaOpticsAI-Optimization")
    register_subset(mcp, controller.tools, TOOLS)
    if args.transport == "stdio":
        mcp.run()
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()