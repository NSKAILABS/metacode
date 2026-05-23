"""Artifact store — pluggable handle-based storage."""
from metaopticsai.store.base import ArtifactStore
from metaopticsai.store.memory import InMemoryArtifactStore

__all__ = ["ArtifactStore", "InMemoryArtifactStore"]