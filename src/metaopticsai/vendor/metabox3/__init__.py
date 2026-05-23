"""Vendored metabox3 (TF-based RCWA simulator).

Copy your existing metabox3/ folder here verbatim. Only
`physics/backends/metabox.py` should import from this package.
"""

# Re-export the canonical submodules — at runtime these only resolve once
# the actual metabox3 sources are copied into this directory.
try:
    from . import rcwa, utils, modeling, assembly  # noqa: F401
except ImportError as _e:  # pragma: no cover
    import warnings
    warnings.warn(
        f"vendor.metabox3 not populated yet ({_e}); copy your metabox3/ "
        "directory into src/metaopticsai/vendor/metabox3/."
    )
