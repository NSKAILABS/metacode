"""matplotlib helpers for rendering tool outputs as ImageContent.

Imports matplotlib lazily and uses the Agg backend — safe for tunneled MCP
servers and headless workers. Never touches pyplot's global figure stack.
"""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np


def _agg_pyplot():
    """Lazy matplotlib import with Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_to_png_b64(fig, dpi: int = 110) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    plt = _agg_pyplot()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def array_to_png_b64(
    arr: np.ndarray,
    *,
    title: str = "",
    cmap: str = "twilight",
    log_scale: bool = False,
    figsize: tuple[float, float] = (5, 5),
) -> str:
    """Render a 2D array to a base64 PNG."""
    plt = _agg_pyplot()
    fig, ax = plt.subplots(figsize=figsize)

    if log_scale:
        from matplotlib.colors import LogNorm
        vmin = max(float(arr.min()), 1e-12)
        im = ax.imshow(arr, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=float(arr.max())))
    else:
        im = ax.imshow(arr, cmap=cmap)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig_to_png_b64(fig)


def chart_to_png_b64(plot_fn, *args, figsize=(6, 4), **kwargs) -> str:
    """Build a chart via `plot_fn(ax, *args, **kwargs)` and return PNG."""
    plt = _agg_pyplot()
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax, *args, **kwargs)
    fig.tight_layout()
    return fig_to_png_b64(fig)