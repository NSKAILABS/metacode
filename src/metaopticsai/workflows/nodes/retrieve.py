"""Retrieve node — pulls relevant background docs from the knowledge base."""
from __future__ import annotations

from typing import Callable

from metaopticsai.rag.base import KnowledgeBase
from metaopticsai.workflows.state import AutoMLState


def make_retrieve_node(kb: KnowledgeBase, k: int = 4) -> Callable[[AutoMLState], dict]:
    def retrieve_node(state: AutoMLState) -> dict:
        parsed = state.get("parsed", {})
        query = (
            f"metalens at {parsed.get('wavelength_nm', '?')} nm, "
            f"NA derived from D={parsed.get('diameter_um')} um and "
            f"f={parsed.get('focal_length_um')} um, "
            f"material={parsed.get('preferred_material')}, "
            f"polarization={parsed.get('polarization')}"
        )
        docs = kb.retrieve(query, k=k)
        return {
            "retrieved_docs": [
                {"content": d.content, "source": d.source, "score": d.score}
                for d in docs
            ],
            "retrieval_iter": state.get("retrieval_iter", 0) + 1,
        }

    return retrieve_node