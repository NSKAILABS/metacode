"""FAISS-backed knowledge base with keyword-search fallback.

The Self-RAG workflow has a `grade_documents` step downstream that drops
low-quality retrievals, so a slightly noisy retriever is fine here.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from metaopticsai.llm.base import LLMProvider
from metaopticsai.rag.base import KnowledgeBase, RetrievedDoc
from metaopticsai.rag.loader import load_corpus

log = logging.getLogger(__name__)


class FAISSKnowledgeBase(KnowledgeBase):
    """In-memory FAISS index over the markdown corpus.

    If the embedder is unavailable, falls back to naive substring scoring.
    """

    def __init__(
        self,
        corpus_dir: Path | str,
        embeddings_provider: LLMProvider,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.embeddings_provider = embeddings_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._vs: FAISS | None = None
        self._docs: dict[str, str] = {}
        self._build()

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        self._docs = load_corpus(self.corpus_dir)
        if not self._docs:
            log.warning("Empty corpus — knowledge base will return nothing.")
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: list[Document] = []
        for stem, content in self._docs.items():
            for i, chunk in enumerate(splitter.split_text(content)):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={"source": stem, "chunk": i},
                ))

        if not self.embeddings_provider.supports_embeddings:
            log.warning("Provider %s lacks embeddings — semantic retrieval disabled.",
                        self.embeddings_provider.name)
            return
        try:
            emb = self.embeddings_provider.embeddings()
            self._vs = FAISS.from_documents(chunks, emb)
            log.info("Built FAISS index over %d chunks from %d documents",
                     len(chunks), len(self._docs))
        except Exception as e:
            log.warning("FAISS build failed: %s — falling back to keyword search.", e)
            self._vs = None

    # ------------------------------------------------------------ retrieval

    def retrieve(self, query: str, *, k: int = 4) -> list[RetrievedDoc]:
        if self._vs is not None:
            results = self._vs.similarity_search_with_score(query, k=k)
            return [
                RetrievedDoc(
                    content=d.page_content,
                    source=d.metadata.get("source", "unknown"),
                    score=float(score),
                    metadata=d.metadata,
                )
                for d, score in results
            ]
        # Fallback: rank docs by # of overlapping query terms.
        q_terms = {t.lower() for t in query.split() if len(t) > 3}
        ranked: list[tuple[float, str, str]] = []
        for stem, content in self._docs.items():
            text_l = content.lower()
            score = sum(1 for t in q_terms if t in text_l)
            if score > 0:
                ranked.append((float(score), stem, content[:1500]))
        ranked.sort(key=lambda r: r[0], reverse=True)
        return [
            RetrievedDoc(content=c, source=s, score=sc)
            for sc, s, c in ranked[:k]
        ]

    def is_ready(self) -> bool:
        return bool(self._docs)