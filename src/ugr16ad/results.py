"""One file format for every model's results, so charts never depend on model code.

Each run writes to ``reports/results/<model>/seed<seed>/``:

- ``scores.csv``: one row per scored minute; columns ``timestamp, part, score``.
- ``meta.json``: at least ``model``, ``seed`` and ``threshold``, plus anything
  else worth reporting (settings, training seconds, parameter count).
- ``latents.npy`` (optional): row i is the embedding of scores row i.
- ``feature_errors.npy`` (optional): row i is the per-feature squared error of
  scores row i, for "which features caused this alert".

Charts and tables read only these files. That lets the charts be built on PCA
in week 2 and rerun unchanged on every other model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ugr16ad.paths import RESULTS_DIR

SCORE_COLUMNS = ("timestamp", "part", "score")
REQUIRED_META = ("model", "seed", "threshold")


@dataclass(frozen=True)
class Run:
    """A loaded run. ``latents`` and ``feature_errors`` are None when not saved."""

    scores: pd.DataFrame
    meta: dict
    latents: np.ndarray | None
    feature_errors: np.ndarray | None


def run_dir(model: str, seed: int, root: Path = RESULTS_DIR) -> Path:
    """``root / model / f"seed{seed}"``. Does not create it."""
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")


def save_run(
    directory: Path,
    *,
    scores: pd.DataFrame,
    meta: dict,
    latents: np.ndarray | None = None,
    feature_errors: np.ndarray | None = None,
) -> None:
    """Write a run in the format above, creating ``directory`` if needed.

    Raises:
        ValueError: if ``scores`` lacks a ``SCORE_COLUMNS`` column, ``meta``
            lacks a ``REQUIRED_META`` key, or an optional array's row count
            differs from ``len(scores)``. Bad files caught here would otherwise
            surface as a wrong chart two weeks later.
    """
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")


def load_run(directory: Path) -> Run:
    """Read a run written by ``save_run``; ``timestamp`` comes back as datetime."""
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")
