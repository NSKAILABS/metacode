"""Register MCP resources and prompts on a FastMCP instance.

Exposes the on-disk knowledge corpus as `photonics://kb/<source>` resources
plus a few canonical prompts (workflow guide, materials cheatsheet).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from metaopticsai.rag.loader import load_corpus

log = logging.getLogger(__name__)


def register_resources(mcp: Any, corpus_dir: Path | str) -> int:
    """Register every markdown in `corpus_dir` as a photonics:// resource."""
    corpus = load_corpus(corpus_dir)
    for stem, content in corpus.items():
        uri = f"photonics://kb/{stem}"

        # Bind locals via default args to avoid late-binding bug in the loop.
        @mcp.resource(uri)
        async def _resource(content=content) -> str:  # noqa: B008
            return content

        # FastMCP uses the function name; give each a unique one.
        _resource.__name__ = f"kb_{stem}"

    log.info("Registered %d MCP knowledge-base resources.", len(corpus))
    return len(corpus)


def register_prompts(mcp: Any) -> None:
    """Register canonical MCP prompts (templates a client can invoke)."""

    @mcp.prompt(name="design_a_metalens")
    async def design_a_metalens_prompt(
        wavelength_nm: int = 532,
        focal_length_um: float = 100.0,
        diameter_um: float = 50.0,
        target_strehl: float = 0.85,
    ) -> str:
        return (
            f"Design a metalens for {wavelength_nm} nm wavelength with "
            f"focal length {focal_length_um} µm, diameter {diameter_um} µm, "
            f"and Strehl ratio ≥ {target_strehl}.\n\n"
            f"Follow this workflow:\n"
            f"  1. list_materials → choose a suitable material for the band.\n"
            f"  2. run_rcwa_sweep with that material.\n"
            f"  3. generate_phase_mask (mask_type=fzl).\n"
            f"  4. analyze_psf → analyze_mtf → zernike_decompose.\n"
            f"  5. optimize_metalens (heuristic).\n"
            f"  6. export_gds.\n"
            f"Report intermediate handles and metrics at each step."
        )