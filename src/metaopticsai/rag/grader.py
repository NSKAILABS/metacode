"""Self-RAG document grader. Given (query, doc), return a relevance score 0-1."""
from __future__ import annotations

import json
import logging

from langchain_core.prompts import PromptTemplate

from metaopticsai.llm.base import LLMProvider
from metaopticsai.rag.base import RetrievedDoc
from metaopticsai.utils.json_io import safe_parse_llm_json

log = logging.getLogger(__name__)

_GRADE_PROMPT = PromptTemplate.from_template(
    """You grade the relevance of a retrieved document to a metalens-design query.

Query: {query}

Document (truncated): {doc}

Respond with ONLY a JSON object: {{"score": <0-1>, "reason": "<short>"}}

Use 1.0 for highly relevant, 0.0 for unrelated. No prose outside JSON.
"""
)


class DocumentGrader:
    """Filters retrieved documents using an LLM judge."""

    def __init__(self, provider: LLMProvider, threshold: float = 0.5):
        self.provider = provider
        self.threshold = threshold
        self._chain = _GRADE_PROMPT | provider.chat_model()

    def grade(self, query: str, doc: RetrievedDoc) -> float:
        try:
            resp = self._chain.invoke({"query": query, "doc": doc.content[:1000]})
            text = resp.content if hasattr(resp, "content") else str(resp)
            parsed = safe_parse_llm_json(text, default={"score": 0.5})
            return float(parsed.get("score", 0.5))
        except Exception as e:
            log.warning("Grader failed (%s); defaulting to 0.5", e)
            return 0.5

    def filter(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        kept = []
        for d in docs:
            s = self.grade(query, d)
            if s >= self.threshold:
                kept.append(d)
        return kept