"""
MetaBox - A Python package for metasurface design and simulation.

This version is compatible with Python 3.11.9+
"""
from importlib.metadata import PackageNotFoundError, version

try:
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError