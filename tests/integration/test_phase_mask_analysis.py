"""End-to-end test of phase mask → PSF → MTF, no LLM in the loop.

Requires numpy + scipy. Skips if either is missing.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")


def test_fzl_to_psf_to_mtf():
    from metaopticsai.physics.backends.analytical import AnalyticalBackend
    from metaopticsai.physics.backends.base import BackendRegistry
    from metaopticsai.store.memory import InMemoryArtifactStore
    from metaopticsai.tools import build_tool_registry

    store = InMemoryArtifactStore()
    backends = BackendRegistry()
    backends.register(AnalyticalBackend())
    tools = build_tool_registry(store, backends)

    mask_tool = tools.get("generate_phase_mask")
    psf_tool = tools.get("analyze_psf")
    mtf_tool = tools.get("analyze_mtf")

    async def _run():
        mask = await mask_tool.call({
            "mask_type": "fzl",
            "wavelength_nm": 532,
            "diameter_um": 50.0,
            "focal_length_um": 100.0,
            "pixel_size_um": 0.35,
        })
        psf = await psf_tool.call({
            "mask_handle": mask["handle"],
            "wavelength_nm": 532,
            "pixel_size_um": 0.35,
            "focal_length_um": 100.0,
        })
        mtf = await mtf_tool.call({
            "psf_handle": psf["handle"],
            "pixel_size_um": 0.35,
            "wavelength_nm": 532,
            "focal_length_um": 100.0,
        })
        return mask, psf, mtf

    mask, psf, mtf = asyncio.run(_run())
    assert "handle" in mask
    assert 0.0 < psf["strehl_ratio"] <= 1.01
    assert mtf["mtf50_lpmm"] >= 0.0