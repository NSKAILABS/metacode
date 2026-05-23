"""ArtifactStore — abstract handle store. Implementations: in-memory, Redis, SQLite."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from metaopticsai.domain.handles import Artifact, ArtifactKind


class ArtifactStore(ABC):
    """Pluggable store for large arrays referenced by short handle.

    All implementations must be thread-safe.
    Tool `_run` methods may run inside a thread pool, so `put`/`get` are sync;
    if a backend is async, wrap with `asyncio.to_thread`.
    """

    @abstractmethod
    def put(self, kind: ArtifactKind, payload: Any, **metadata: Any) -> str:
        """Store an artifact; return its handle."""

    @abstractmethod
    def get(self, handle: str, expected_kind: ArtifactKind | None = None) -> Artifact:
        """Retrieve. Raises KeyError on miss or kind mismatch."""

    @abstractmethod
    def list(self, kind: ArtifactKind | None = None) -> list[dict[str, Any]]:
        """List summaries, newest first."""

    @abstractmethod
    def drop(self, handle: str) -> bool:
        """Remove. Returns True iff existed."""

    @abstractmethod
    def clear(self) -> int:
        """Remove all. Returns count removed."""