"""MetaOpticsController — the single composition root.

Wires together every layer of the stack:
  - ArtifactStore + BackendRegistry + JobManager
  - Tool layer
  - LLM provider + RAG KB + grader
  - LangGraph workflow

Used by:
  - the MCP servers (they expose `controller.tools` and `controller.design_metalens(...)`)
  - the headless workflow CLI (`scripts/run_workflow.py`)
  - notebooks and tests
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from metaopticsai.configs.settings import settings
from metaopticsai.llm.base import LLMProvider
from metaopticsai.llm.registry import get_provider
from metaopticsai.physics.backends import default_registry
from metaopticsai.physics.backends.base import BackendRegistry
from metaopticsai.orchestration.job_manager import JobManager
from metaopticsai.rag.base import KnowledgeBase
from metaopticsai.rag.faiss_kb import FAISSKnowledgeBase
from metaopticsai.rag.grader import DocumentGrader
from metaopticsai.store.base import ArtifactStore
from metaopticsai.store.memory import InMemoryArtifactStore
from metaopticsai.tools import build_tool_registry
from metaopticsai.tools.base import ToolRegistry
from metaopticsai.workflows import build_metalens_self_rag_graph

log = logging.getLogger(__name__)


@dataclass
class MetaOpticsController:
    """High-level controller. Build with `MetaOpticsController.build(...)`."""

    store: ArtifactStore
    backends: BackendRegistry
    provider: LLMProvider
    kb: KnowledgeBase
    grader: DocumentGrader
    tools: ToolRegistry
    job_manager: JobManager
    _graph: Any = field(default=None)

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        provider_name: str | None = None,
        corpus_dir: Path | str | None = None,
    ) -> "MetaOpticsController":
        store = InMemoryArtifactStore()
        backends = default_registry()
        job_manager = JobManager()
        provider = get_provider(provider_name)

        # Resolve corpus path
        cdir = Path(corpus_dir) if corpus_dir else (
            Path(__file__).resolve().parent.parent / "rag" / "corpus"
        )
        kb = FAISSKnowledgeBase(cdir, embeddings_provider=provider)
        grader = DocumentGrader(provider, threshold=settings.workflow.grader_threshold)

        # Build tools — controller_factory uses a fresh build of *this* class
        # (lazy so we don't recurse).
        def _controller_factory() -> "MetaOpticsController":
            return cls.build(provider_name=provider_name, corpus_dir=corpus_dir)

        tools = build_tool_registry(
            store=store,
            backends=backends,
            job_manager=job_manager,
            controller_factory=_controller_factory,
        )

        return cls(
            store=store, backends=backends, provider=provider,
            kb=kb, grader=grader, tools=tools, job_manager=job_manager,
        )

    # ── workflow ──────────────────────────────────────────────────────────

    def graph(self):
        """Lazily compile the Self-RAG graph once."""
        if self._graph is None:
            self._graph = build_metalens_self_rag_graph(
                self.provider, self.kb, self.grader, self.tools,
            )
        return self._graph

    async def design_metalens(
        self, requirement: str, *, target_strehl: float | None = None,
    ) -> dict:
        """End-to-end metalens design from a natural-language requirement."""
        state = {
            "requirement": requirement,
            "target_strehl": target_strehl or settings.workflow.default_target_strehl,
        }
        log.info("Starting AutoML design — requirement=%r", requirement[:120])
        result = await self.graph().ainvoke(state)
        return result