"""Retrieval-augmented knowledge base for metalens design."""
from metaopticsai.rag.base import KnowledgeBase, RetrievedDoc
from metaopticsai.rag.faiss_kb import FAISSKnowledgeBase
from metaopticsai.rag.grader import DocumentGrader
from metaopticsai.rag.loader import load_corpus

__all__ = [
    "KnowledgeBase", "RetrievedDoc",
    "FAISSKnowledgeBase", "DocumentGrader",
    "load_corpus",
]