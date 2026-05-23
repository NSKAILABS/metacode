"""Knowledge base abstraction. Concrete impls: FAISSKnowledgeBase (semantic)
and a future keyword-only fallback."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    content: str
    source: str
    score: float = 0.0
    metadata: dict | None = None


class KnowledgeBase(ABC):
    """Abstract retrieval interface used by Self-RAG nodes."""

    @abstractmethod
    def retrieve(self, query: str, *, k: int = 4) -> list[RetrievedDoc]: ...

    @abstractmethod
    def is_ready(self) -> bool: ...