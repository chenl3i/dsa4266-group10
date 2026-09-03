"""Uniform logging setup.

Scripts call :func:`setup_logging` once; library modules only ever call
``logging.getLogger(__name__)`` so that log configuration stays the caller's
decision.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging for a script or notebook session."""
    logging.basicConfig(
        level=level,
        format=_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # The requests/urllib3 stack is noisy during NVD paging.
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
