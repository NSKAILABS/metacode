"""Orchestration layer — wires every component together.

`JobManager` is dependency-light (stdlib only). `MetaOpticsController` is
heavier and pulls in the full LLM + LangGraph stack, so it's lazy-loaded via
__getattr__ to keep `from metaopticsai.orchestration import JobManager`
cheap.
"""
from __future__ import annotations

from metaopticsai.orchestration.job_manager import Job, JobManager

__all__ = ["JobManager", "Job", "MetaOpticsController"]


def __getattr__(name: str):
    if name == "MetaOpticsController":
        from metaopticsai.orchestration.controller import MetaOpticsController
        return MetaOpticsController
    raise AttributeError(f"module 'metaopticsai.orchestration' has no attribute {name!r}")
