"""Shared utilities for FastMCP server processes."""
from metaopticsai.mcp_servers.common.adapter import (
    register_tool_on_mcp, register_all,
)
from metaopticsai.mcp_servers.common.resources import (
    register_resources, register_prompts,
)

__all__ = [
    "register_tool_on_mcp", "register_all",
    "register_resources", "register_prompts",
]
