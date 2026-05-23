"""Build (or rebuild) the FAISS knowledge base and print a summary.

    python scripts/seed_knowledge_base.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaopticsai.llm.registry import get_provider  # noqa: E402
from metaopticsai.rag.faiss_kb import FAISSKnowledgeBase  # noqa: E402
from metaopticsai.utils.logging import setup_logging  # noqa: E402


def main() -> None:
    setup_logging()
    provider = get_provider()
    corpus_dir = Path(__file__).resolve().parents[1] / "src" / "metaopticsai" / "rag" / "corpus"
    kb = FAISSKnowledgeBase(corpus_dir, embeddings_provider=provider)
    print(f"KB ready: {kb.is_ready()}  (provider={provider.name})")
    docs = kb.retrieve("metalens phase coverage 532 nm TiO2", k=3)
    for i, d in enumerate(docs):
        print(f"\n[{i}] source={d.source} score={d.score:.3f}")
        print(d.content[:300])


if __name__ == "__main__":
    main()