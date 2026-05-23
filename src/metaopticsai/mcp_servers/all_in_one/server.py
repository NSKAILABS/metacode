"""All-in-one MCP server — exposes the full tool catalog over stdio or HTTP.

Use:
    python -m metaopticsai.mcp_servers.all_in_one.server --transport stdio
    python -m metaopticsai.mcp_servers.all_in_one.server --transport http --port 8765
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fastmcp import FastMCP

from metaopticsai.configs.settings import settings
from metaopticsai.mcp_servers.common import (
    register_all, register_resources, register_prompts,
)
from metaopticsai.orchestration.controller import MetaOpticsController
from metaopticsai.utils.logging import setup_logging

log = logging.getLogger(__name__)


def build_server(
    provider_name: str | None = None,
    corpus_dir: Path | str | None = None,
) -> tuple[FastMCP, MetaOpticsController]:
    """Build a FastMCP server and the controller it wraps."""
    controller = MetaOpticsController.build(
        provider_name=provider_name, corpus_dir=corpus_dir,
    )
    mcp = FastMCP(settings.server.name)
    register_all(mcp, controller.tools)
    register_resources(mcp, corpus_dir or Path(__file__).resolve().parents[2] / "rag" / "corpus")
    register_prompts(mcp)
    log.info("Server %r built with %d tools.", settings.server.name, len(controller.tools))
    return mcp, controller


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="MetaOpticsAI all-in-one MCP server")
    p.add_argument("--transport", choices=("stdio", "http", "sse"), default="stdio")
    p.add_argument("--host", default=settings.server.host)
    p.add_argument("--port", type=int, default=settings.server.port)
    p.add_argument("--provider", default=None, help="LLM provider name")
    p.add_argument("--corpus", default=None, help="Override corpus dir")
    args = p.parse_args()

    mcp, _ = build_server(provider_name=args.provider, corpus_dir=args.corpus)
    log.info("Starting MCP on transport=%s host=%s port=%s",
             args.transport, args.host, args.port)

    if args.transport == "stdio":
        mcp.run()
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:  # http
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()