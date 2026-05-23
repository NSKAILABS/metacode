"""In-process ArtifactStore. Thread-safe via RLock. Default for single-tenant deploys."""
from __future__ import annotations

import threading
from typing import Any

from metaopticsai.domain.handles import Artifact, ArtifactKind
from metaopticsai.store.base import ArtifactStore


class InMemoryArtifactStore(ArtifactStore):
    """Process-local dict-backed store.

    Trivial to swap for RedisArtifactStore later — all callers go through the
    ABC.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._items: dict[str, Artifact] = {}

    def put(self, kind: ArtifactKind, payload: Any, **metadata: Any) -> str:
        art = Artifact.new(kind, payload, **metadata)
        with self._lock:
            self._items[art.handle] = art
        return art.handle

    def get(self, handle: str, expected_kind: ArtifactKind | None = None) -> Artifact:
        with self._lock:
            art = self._items.get(handle)
        if art is None:
            raise KeyError(f"No artifact with handle {handle!r}")
        if expected_kind is not None and art.kind != expected_kind:
            raise KeyError(
                f"Handle {handle!r} is a {art.kind!r}, expected {expected_kind!r}"
            )
        return art

    def list(self, kind: ArtifactKind | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._items.values())
        if kind is not None:
            items = [a for a in items if a.kind == kind]
        items.sort(key=lambda a: a.created_at, reverse=True)
        return [a.summary() for a in items]

    def drop(self, handle: str) -> bool:
        with self._lock:
            return self._items.pop(handle, None) is not None

    def clear(self) -> int:
        with self._lock:
            n = len(self._items)
            self._items.clear()
        return n