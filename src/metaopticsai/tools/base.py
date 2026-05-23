"""BaseTool + ToolRegistry — the transport-agnostic scientific tool API.

A `BaseTool`:
  - declares Pydantic `Input` and (optionally) `Output` classes;
  - implements `_run(params)` — sync OR async, returning Output, dict, or MCP
    content blocks (list of TextContent/ImageContent);
  - is callable directly: `await tool.call({...})`;
  - is wrappable into a FastMCP `@mcp.tool` via `mcp_servers/common/adapter.py`.

Dependencies (store, backends, LLM provider) are *injected* in `__init__`.
There are no module-level globals.
"""
from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterator, TypeVar

from pydantic import BaseModel

from metaopticsai.store.base import ArtifactStore

log = logging.getLogger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseTool(ABC, Generic[InputT, OutputT]):
    """Abstract scientific tool with typed schemas."""

    # Subclasses MUST set these:
    name: str
    description: str
    Input: type[BaseModel]
    Output: type[BaseModel] | None = None  # None → free-form dict or content blocks

    # Optional flags:
    needs_store: bool = False

    def __init__(self, store: ArtifactStore | None = None):
        if self.needs_store and store is None:
            raise ValueError(f"{self.name} requires an ArtifactStore.")
        self.store = store

    # ── Validate-call-validate wrapper ─────────────────────────────────────

    async def call(self, raw_input: dict | BaseModel) -> Any:
        """Validate input, run, validate output. Returns:

          - the Output model's dict if Output is declared;
          - the raw return value (dict, list of content blocks) otherwise.
        """
        if isinstance(raw_input, BaseModel):
            inp = raw_input
        else:
            inp = self.Input.model_validate(raw_input)

        if log.isEnabledFor(logging.DEBUG):
            log.debug("→ %s(%s)", self.name, inp.model_dump_json()[:300])

        out = self._run(inp)
        if inspect.iscoroutine(out):
            out = await out

        # Output validation only if Output is declared and the return is
        # something that looks like a dict/model — content-block lists pass
        # through unchanged.
        if self.Output is not None:
            if isinstance(out, BaseModel):
                return out.model_dump()
            if isinstance(out, dict):
                return self.Output.model_validate(out).model_dump()
        return out

    def call_sync(self, raw_input: dict | BaseModel) -> Any:
        """Synchronous wrapper for sync callers (e.g. LangGraph nodes).

        If `_run` is async, this blocks via asyncio.run; only safe outside
        an existing event loop.
        """
        import asyncio
        if isinstance(raw_input, BaseModel):
            inp = raw_input
        else:
            inp = self.Input.model_validate(raw_input)

        out = self._run(inp)
        if inspect.iscoroutine(out):
            out = asyncio.run(out)

        if self.Output is not None:
            if isinstance(out, BaseModel):
                return out.model_dump()
            if isinstance(out, dict):
                return self.Output.model_validate(out).model_dump()
        return out

    # ── Subclass hook ──────────────────────────────────────────────────────

    @abstractmethod
    def _run(self, params: InputT) -> OutputT | dict | list | Any:
        """Sync or async. Return Output instance, dict, or content blocks."""


class ToolRegistry:
    """Holds a set of BaseTools keyed by name. Wrapped by the MCP adapter
    and consumed directly by LangGraph nodes."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool {name!r}. Known: {list(self._tools)}")
        return self._tools[name]

    def list(self) -> list[BaseTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)