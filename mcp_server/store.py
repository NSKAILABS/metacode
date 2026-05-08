"""
mcp_server.store — In-process session store for MetaOpticsAI MCP tools.

Tools that produce big artefacts (phase masks, RCWA sweep tables, PSF arrays)
return a short UUID handle to the LLM rather than the full array. The LLM
then passes that handle to downstream tools. This keeps the model's context
window small and predictable, and makes the tools cheaply composable.

The store is a process-local dict — fine for single-user desktop and
single-tenant FastAPI use. For multi-tenant or persistent deployments, swap
the dict for SQLite or Redis without changing tool signatures.
"""
from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal


# ── Public types ───────────────────────────────────────────────────────────

ArtifactKind = Literal[
    "phase_mask",        # 2-D ndarray of phase (radians)
    "quantized_mask",    # 2-D ndarray of int levels + bin edges
    "rcwa_sweep",        # dict: diameters_nm, amplitudes, phases_rad
    "psf",               # dict: psf_2d, x_um, y_um
    "mtf",               # dict: freqs_lpmm, mtf_radial, mtf_diff_limit
    "zernike",           # dict: coefficients (Noll index → coeff), names
    "gds_file",          # Path to a written .gds on disk
    "metalens_design",   # complete design state (DesignState)
    "job_result",        # opaque dict from a long-running job
]


@dataclasses.dataclass
class Artifact:
    """A single artefact stored under a handle."""
    handle: str
    kind: ArtifactKind
    payload: Any
    created_at: float
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Cheap, LLM-safe description (no big arrays)."""
        s = {
            "handle": self.handle,
            "kind": self.kind,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
        # Try to add a useful "shape" hint without leaking the array
        p = self.payload
        try:
            if hasattr(p, "shape"):
                s["shape"] = list(p.shape)
            elif isinstance(p, dict):
                s["keys"] = list(p.keys())
            elif isinstance(p, Path):
                s["path"] = str(p)
                if p.exists():
                    s["size_kb"] = round(p.stat().st_size / 1024, 1)
        except Exception:
            pass
        return s


# ── Module-level store ─────────────────────────────────────────────────────

_lock = threading.Lock()
_store: dict[str, Artifact] = {}


def put(kind: ArtifactKind, payload: Any, **metadata: Any) -> str:
    """Store an artefact and return its handle."""
    handle = uuid.uuid4().hex[:12]   # 12 hex chars is plenty
    with _lock:
        _store[handle] = Artifact(
            handle=handle,
            kind=kind,
            payload=payload,
            created_at=time.time(),
            metadata=dict(metadata),
        )
    return handle


def get(handle: str, expected_kind: ArtifactKind | None = None) -> Artifact:
    """Fetch an artefact by handle. Raises KeyError if missing or wrong kind."""
    with _lock:
        a = _store.get(handle)
    if a is None:
        raise KeyError(f"No artefact with handle {handle!r}")
    if expected_kind is not None and a.kind != expected_kind:
        raise KeyError(
            f"Handle {handle!r} is a {a.kind!r}, not {expected_kind!r}"
        )
    return a


def list_handles(kind: ArtifactKind | None = None) -> list[dict[str, Any]]:
    """List summaries of all stored artefacts, optionally filtered by kind."""
    with _lock:
        items = list(_store.values())
    if kind is not None:
        items = [a for a in items if a.kind == kind]
    items.sort(key=lambda a: a.created_at, reverse=True)
    return [a.summary() for a in items]


def drop(handle: str) -> bool:
    """Remove an artefact. Returns True if it existed."""
    with _lock:
        return _store.pop(handle, None) is not None


def clear() -> int:
    """Drop everything. Returns the count removed."""
    with _lock:
        n = len(_store)
        _store.clear()
    return n
