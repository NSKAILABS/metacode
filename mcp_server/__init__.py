from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path

def _ensure_metaopticsai_importable() -> None:
    try:
        import metaopticsai  
        return
    except ImportError:
        pass
    src = _Path(__file__).resolve().parents[1] / "src"
    if src.is_dir() and str(src) not in _sys.path:
        _sys.path.insert(0, str(src))


_ensure_metaopticsai_importable()
