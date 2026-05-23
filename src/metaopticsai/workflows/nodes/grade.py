"""Grade node — LLM filters retrieved docs by relevance (Self-RAG step)."""
from __future__ import annotations

from typing import Callable

from metaopticsai.rag.base import RetrievedDoc
from metaopticsai.rag.grader import DocumentGrader
from metaopticsai.workflows.state import AutoMLState


def make_grade_node(grader: DocumentGrader) -> Callable[[AutoMLState], dict]:
    def grade_node(state: AutoMLState) -> dict:
        parsed = state.get("parsed", {})
        query = f"metalens at {parsed.get('wavelength_nm')} nm material={parsed.get('preferred_material')}"
        raw = state.get("retrieved_docs", [])
        docs = [
            RetrievedDoc(content=d["content"], source=d["source"], score=d.get("score", 0.0))
            for d in raw
        ]
        kept = grader.filter(query, docs)
        return {
            "retrieved_docs": [
                {"content": d.content, "source": d.source, "score": d.score}
                for d in kept
            ],
        }

    return grade_node