"""Optimizer abstraction. Subclasses implement different strategies for
driving an FOM toward a target."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.results import FOMResult


class Optimizer(ABC):
    """Common interface for design-space optimizers.

    The concrete strategies are:
      - HeuristicOptimizer: analytical FOM, no gradients, very fast.
      - GradientOptimizer:  TF gradient tape through metabox3.
      - SurrogateOptimizer: NN surrogate-based, requires a trained metamodel.
    """

    name: str

    @abstractmethod
    def optimize(
        self,
        params: MetalensDesignParams,
        *,
        target_fom: float,
        max_iters: int,
        **kwargs: Any,
    ) -> tuple[MetalensDesignParams, FOMResult, list[float]]:
        """Returns (final_params, final_fom, fom_history)."""