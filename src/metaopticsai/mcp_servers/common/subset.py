"""Register only a named subset of tools onto a FastMCP server. Used by the
domain-specific servers (rcwa_server, optimization_server, etc.)."""
from __future__ import annotations

import logging
from typing import Any, Iterable

from metaopticsai.tools.base import ToolRegistry
from metaopticsai.mcp_servers.common.adapter import register_tool_on_mcp

log = logging.getLogger(__name__)


def register_subset(mcp: Any, registry: ToolRegistry, names: Iterable[str]) -> int:
    n = 0
    for name in names:
        try:
            tool = registry.get(name)
        except KeyError:
            log.warning("Subset includes unknown tool %r — skipping.", name)
            continue
        register_tool_on_mcp(mcp, tool)
        n += 1
    log.info("Registered subset of %d tools.", n)
    return n