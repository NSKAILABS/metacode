"""Conditional edges used by the metalens Self-RAG graph.

Each predicate inspects `AutoMLState` and returns the name of the next
node — that name MUST match a node registered in graphs/metalens_self_rag.py.
"""
from __future__ import annotations

from metaopticsai.configs.settings import settings
from metaopticsai.workflows.state import AutoMLState


def route_after_grade(state: AutoMLState) -> str:
    """If grader kept enough docs, generate; otherwise retrieve again."""
    docs = state.get("retrieved_docs", [])
    iters = state.get("retrieval_iter", 0)
    if len(docs) >= settings.workflow.min_kept_docs:
        return "generate_params"
    if iters >= settings.workflow.max_retrieval_iters:
        return "generate_params"   # give up gracefully, don't loop forever
    return "retrieve"


def route_after_reflect(state: AutoMLState) -> str:
    """Self-RAG branching point after the reflection node."""
    decision = state.get("decision", "accept")
    if decision == "accept":
        return "finalize"
    if decision == "re_retrieve":
        if state.get("retrieval_iter", 0) >= settings.workflow.max_retrieval_iters:
            return "finalize"
        return "retrieve"
    if decision == "re_sweep":
        return "simulate_rcwa"
    if decision == "re_optimize":
        if state.get("optimization_iter", 0) >= settings.workflow.max_optim_iters:
            return "finalize"
        return "optimize"
    # unknown decision → fail safe
    return "finalize"