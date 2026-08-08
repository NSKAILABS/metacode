"""MetaOpticsAI — computational nanophotonics platform.

Core physics, phase, optimization, and analysis modules are importable directly:

    from metaopticsai.physics.backends.metabox import MetaboxBackend
    from metaopticsai.optimization.gradient import GradientOptimizer

Follow the dependency rules in pyproject.toml [tool.importlinter].
"""

__version__ = "0.2.0"
__all__ = ["__version__"]
