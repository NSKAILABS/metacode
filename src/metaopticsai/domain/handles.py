"""Artifact types — the value object passed between tools by handle.

This module is a *pure leaf*: it must not import from anywhere else in
metaopticsai. All it knows is how to wrap a payload with a UUID and metadata.
"""
from __future__ import annotations

import dataclasses
import time
import uuid
from pathlib import Path
from typing import Any, Literal


ArtifactKind = Literal[
    "phase_mask",          # 2D ndarray of phase (radians)
    "quantized_mask",      # 2D ndarray of integer levels + bin edges
    "rcwa_sweep",          # dict: diameters_nm, amplitudes, phases_rad, …
    "psf",                 # dict: psf_2d, x_um, y_um
    "mtf",                 # dict: freqs_lpmm, mtf_radial, mtf_diff_limit
    "zernike",             # dict: coefficients (Noll → coeff), names
    "field_2d",            # 2D intensity / complex field
    "gds_file",            # Path to a written .gds on disk
    "metalens_design",     # Complete design dict
    "metamodel",           # Path to a trained metamodel directory
    "job_result",          # Opaque dict from a long-running job
]


@dataclasses.dataclass(frozen=True, slots=True)
class Artifact:
    """Immutable wrapper around a stored payload.

    `handle` is a 12-char hex UUID; treat as opaque.
    `payload` is the actual value (ndarray, dict, Path, …) — kept on the server
    side. Never serialize the payload into LLM context; pass the handle instead.
    """

    handle: str
    kind: ArtifactKind
    payload: Any
    created_at: float
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    # ─ constructors ──────────────────────────────────────────────────────────

    @classmethod
    def new(cls, kind: ArtifactKind, payload: Any, **metadata: Any) -> "Artifact":
        return cls(
            handle=uuid.uuid4().hex[:12],
            kind=kind,
            payload=payload,
            created_at=time.time(),
            metadata=dict(metadata),
        )

    # ─ LLM-safe summary ──────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """A JSON-safe description suitable for sending to an LLM.

        Crucially, this never includes the payload bytes — only shape/keys/path.
        """
        s: dict[str, Any] = {
            "handle": self.handle,
            "kind": self.kind,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }
        p = self.payload
        try:
            if hasattr(p, "shape"):
                s["shape"] = list(getattr(p, "shape"))
            elif isinstance(p, dict):
                s["keys"] = list(p.keys())
            elif isinstance(p, Path):
                s["path"] = str(p)
                if p.exists():
                    s["size_kb"] = round(p.stat().st_size / 1024, 1)
        except Exception:  # noqa: BLE001
            # Summaries must never raise — they're called from logging paths.
            pass
        return s