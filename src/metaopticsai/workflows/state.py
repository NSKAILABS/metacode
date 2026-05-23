"""Shared state for the metalens Self-RAG LangGraph workflow.

Every node receives this dict and returns a partial update. Designed to be
JSON-serializable so we can checkpoint mid-graph if needed.
"""
from __future__ import annotations

from typing import Any, TypedDict


class AutoMLState(TypedDict, total=False):
    # ── input ─────────────────────────────────────────────────────────────
    requirement: str
    target_strehl: float

    # ── parsed requirements (from parse node) ─────────────────────────────
    parsed: dict[str, Any]

    # ── retrieval ─────────────────────────────────────────────────────────
    retrieved_docs: list[dict[str, Any]]    # list of {content, source, score}
    retrieval_iter: int                     # how many times we've retrieved

    # ── design parameters (from generate node) ────────────────────────────
    design_params: dict[str, Any]

    # ── simulation results ────────────────────────────────────────────────
    sweep_handle: str
    sweep_metrics: dict[str, Any]           # {phase_coverage, mean_transmission, ...}

    # ── optimization ──────────────────────────────────────────────────────
    design_handle: str
    achieved_strehl: float
    fom_history: list[float]
    optimization_iter: int                  # how many re-optimizes we've run

    # ── analysis ──────────────────────────────────────────────────────────
    psf_handle: str
    mtf_handle: str

    # ── reflection ────────────────────────────────────────────────────────
    decision: str                           # accept | re_optimize | re_sweep | re_retrieve
    decision_history: list[dict[str, Any]]

    # ── finalization ──────────────────────────────────────────────────────
    gds_handle: str
    gds_path: str
    summary: str

    # ── housekeeping ──────────────────────────────────────────────────────
    errors: list[str]
    warnings: list[str]