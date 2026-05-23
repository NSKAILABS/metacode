"""FastMCP adapter — single bridge between transport-agnostic tools and any
FastMCP server. Auto-generates a `@mcp.tool` from a `BaseTool` so the tool
catalog is the single source of truth.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from metaopticsai.tools.base import BaseTool, ToolRegistry

log = logging.getLogger(__name__)


def register_tool_on_mcp(mcp: Any, tool: BaseTool) -> None:
    """Register a single BaseTool on a FastMCP instance.

    The Pydantic Input class drives FastMCP's argument schema generation —
    every field's `description`, default and constraints flow through to the
    MCP Inspector UI automatically.
    """
    InputSchema = tool.Input

    async def _wrapper(**kwargs: Any) -> Any:
        try:
            return await tool.call(kwargs)
        except Exception as e:  # noqa: BLE001
            log.exception("Tool %s failed", tool.name)
            return {"error": str(e), "tool": tool.name}

    # FastMCP introspects function signatures, so build a signature that
    # matches the Pydantic Input fields. The simplest way is to take a single
    # `params: InputSchema` arg — modern FastMCP supports Pydantic models
    # directly and unfolds them into the MCP tool input schema.
    async def _typed_wrapper(params: InputSchema) -> Any:  # type: ignore[valid-type]
        try:
            return await tool.call(params)
        except Exception as e:  # noqa: BLE001
            log.exception("Tool %s failed", tool.name)
            return {"error": str(e), "tool": tool.name}

    _typed_wrapper.__name__ = tool.name
    _typed_wrapper.__doc__ = tool.description

    mcp.tool(name=tool.name, description=tool.description)(_typed_wrapper)


def register_all(mcp: Any, registry: ToolRegistry) -> int:
    """Register every tool in the registry. Returns the count."""
    n = 0
    for tool in registry:
        register_tool_on_mcp(mcp, tool)
        n += 1
    log.info("Registered %d MCP tools.", n)
    return n