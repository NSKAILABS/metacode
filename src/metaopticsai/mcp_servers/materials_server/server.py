"""Materials-only MCP server — lookup tools."""
from __future__ import annotations

import argparse

from fastmcp import FastMCP

from metaopticsai.configs.settings import settings
from metaopticsai.mcp_servers.common.subset import register_subset
from metaopticsai.orchestration.controller import MetaOpticsController
from metaopticsai.utils.logging import setup_logging

TOOLS = ["list_materials", "get_material_index", "get_fdtd_library"]


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=("stdio", "http", "sse"), default="stdio")
    p.add_argument("--host", default=settings.server.host)
    p.add_argument("--port", type=int, default=settings.server.port + 3)
    args = p.parse_args()

    controller = MetaOpticsController.build()
    mcp = FastMCP("MetaOpticsAI-Materials")
    register_subset(mcp, controller.tools, TOOLS)
    if args.transport == "stdio":
        mcp.run()
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()