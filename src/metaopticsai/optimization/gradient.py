"""Gradient-based optimizer — TensorFlow gradient tape through metabox3.

Builds a `LensAssembly` with a single Metasurface whose meta-atom shape
parameter is marked variable, then runs `optimize_single_lens_assembly`
with FOM = LOG_STREHL_RATIO. Adam is the default optimizer; the LR is
expressed in SI metres because metabox3 stores diameters in metres.
"""
from __future__ import annotations

import logging

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.results import FOMResult
from metaopticsai.optimization.base import Optimizer
from metaopticsai.physics.backends.base import BackendRegistry

log = logging.getLogger(__name__)


class GradientOptimizer(Optimizer):
    """Differentiable optimizer running through the metabox3 assembly module."""

    name = "gradient"

    def __init__(self, backends: BackendRegistry):
        self.backends = backends

    def optimize(
        self,
        params: MetalensDesignParams,
        *,
        target_fom: float,
        max_iters: int,
        learning_rate: float = 2e-9,
        **kwargs,
    ) -> tuple[MetalensDesignParams, FOMResult, list[float]]:
        try:
            backend = self.backends.get("metabox")
            if not backend.is_available():
                raise RuntimeError("metabox backend not available")
        except Exception as e:
            raise RuntimeError(
                "GradientOptimizer requires the metabox backend (TensorFlow)."
            ) from e

        try:
            import tensorflow as tf
            from metaopticsai.vendor.metabox3 import rcwa, utils, assembly
        except ImportError as e:
            raise RuntimeError(
                "GradientOptimizer requires `tensorflow` and the vendored "
                "metabox3 package."
            ) from e

        # ── Build a single-lens assembly with a variable meta-atom diameter ──
        n_index_map = {"TiO2": 2.35, "Si": 3.48, "GaN": 2.32, "SiN": 2.0}
        n_pillar = n_index_map[params.pillar_material]

        diam_feature = utils.Feature(
            vmin=params.min_diameter_nm * 1e-9,
            vmax=params.max_diameter_nm * 1e-9,
            name="diameter",
            sampling=int(kwargs.get("n_sweep", 30)),
        )
        pillar = rcwa.Circle(material=n_pillar, radius=diam_feature)
        layer = rcwa.Layer(
            material=1.0,
            thickness=params.pillar_height_nm * 1e-9,
            shapes=(pillar,),
        )
        period_m = params.periodicity_nm * 1e-9
        uc = rcwa.UnitCell(
            layers=[layer],
            periodicity=(period_m, period_m),
            refl_index=1.0,
            tran_index=params.substrate_index,
        )
        proto = rcwa.ProtoUnitCell(uc)
        incidence = utils.Incidence(wavelength=[params.wavelength_nm * 1e-9])

        # Build a Metasurface with structures marked variable for gradient flow.
        # The exact API may vary by metabox3 minor version — see
        # `vendor.metabox3.assembly` for the constructor in use locally.
        try:
            ms = assembly.Metasurface(
                proto_unit_cell=proto,
                incidence=incidence,
                diameter_m=params.diameter_mm * 1e-3,
                focal_length_m=params.focal_length_mm * 1e-3,
                set_structures_variable=True,
            )
            lens_assembly = assembly.LensAssembly(
                metasurfaces=[ms],
                incidence=incidence,
            )
        except AttributeError as e:
            log.warning(
                "metabox3 assembly API mismatch (%s) — returning unoptimized.", e,
            )
            return params, FOMResult(
                value=0.0, name="strehl_ratio", backend="metabox",
                iteration=0, metadata={"method": self.name, "error": str(e)},
            ), []

        # ── Run the gradient loop ──────────────────────────────────────────
        tf_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        try:
            loss_history = assembly.optimize_single_lens_assembly(
                lens_assembly,
                optimizer=tf_optimizer,
                figure_of_merit=assembly.FigureOfMerit.LOG_STREHL_RATIO,
                n_iter=max_iters,
                verbose=0,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("metabox3 optimization failed")
            return params, FOMResult(
                value=0.0, name="strehl_ratio", backend="metabox",
                iteration=0, metadata={"method": self.name, "error": str(e)},
            ), []

        # Convert log-strehl history → strehl history
        history = [
            float(tf.exp(l).numpy()) if hasattr(l, "numpy") else float(l)
            for l in loss_history
        ]
        final = history[-1] if history else 0.0

        return params, FOMResult(
            value=final,
            name="strehl_ratio",
            backend="metabox",
            iteration=len(history),
            metadata={"method": self.name, "learning_rate": learning_rate},
        ), history