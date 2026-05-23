"""Tool registry composition tests — no external services."""
from __future__ import annotations

import pytest

from metaopticsai.physics.backends.base import BackendRegistry
from metaopticsai.physics.backends.analytical import AnalyticalBackend
from metaopticsai.store.memory import InMemoryArtifactStore
from metaopticsai.tools import build_tool_registry


def _make_backends() -> BackendRegistry:
    """Build a backend registry containing only the analytical backend."""
    reg = BackendRegistry()
    reg.register(AnalyticalBackend())
    return reg


def test_registry_has_core_tools():
    store = InMemoryArtifactStore()
    tools = build_tool_registry(store, _make_backends())
    expected = {
        "run_rcwa_sweep", "get_fdtd_library",
        "list_materials", "get_material_index",
        "generate_phase_mask",
        "analyze_psf", "analyze_mtf", "zernike_decompose",
        "propagate_field",
        "optimize_metalens",
        "export_gds",
        "list_artifacts",
    }
    have = set(tools.names())
    missing = expected - have
    assert not missing, f"missing tools: {missing}"


def test_heavy_tools_only_with_job_manager():
    store = InMemoryArtifactStore()
    tools = build_tool_registry(store, _make_backends())
    assert "train_metamodel" not in tools.names()
    assert "get_job" not in tools.names()

    from metaopticsai.orchestration.job_manager import JobManager
    tools2 = build_tool_registry(store, _make_backends(), job_manager=JobManager())
    assert "train_metamodel" in tools2.names()
    assert "get_job" in tools2.names()