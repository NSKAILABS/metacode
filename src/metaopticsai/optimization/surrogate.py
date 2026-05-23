"""Surrogate-based optimizer — uses a trained metabox3 metamodel for fast
FOM evaluation inside an outer search loop.

Skeleton only; full implementation when needed.
"""
from __future__ import annotations

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.results import FOMResult
from metaopticsai.optimization.base import Optimizer


class SurrogateOptimizer(Optimizer):
    name = "surrogate"

    def __init__(self, metamodel_path: str | None = None):
        self.metamodel_path = metamodel_path

    def optimize(
        self,
        params: MetalensDesignParams,
        *,
        target_fom: float,
        max_iters: int,
        **kwargs,
    ) -> tuple[MetalensDesignParams, FOMResult, list[float]]:
        if self.metamodel_path is None:
            raise RuntimeError(
                "SurrogateOptimizer requires a trained metamodel. Run "
                "`train_metamodel` first."
            )
        # Load metabox3.modeling.load_metamodel(name, save_dir) and run a
        # gradient loop using the surrogate instead of full RCWA.
        return params, FOMResult(
            value=0.0, name="strehl_ratio", backend="surrogate",
            iteration=0, metadata={"method": self.name, "status": "not_implemented"},
        ), []