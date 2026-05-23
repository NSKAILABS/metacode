"""Logging configuration — call `setup_logging()` once at process entry."""
from __future__ import annotations

import logging
import os
import sys


_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s %(message)s"


def setup_logging(level: int | str = logging.INFO, *, force: bool = False) -> None:
    """Idempotent root-logger setup. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # TensorFlow can be exceptionally chatty; honour env or default to ERROR.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    for noisy in ("tensorflow", "absl", "h5py", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor — ensures setup_logging has been called."""
    setup_logging()
    return logging.getLogger(name)
