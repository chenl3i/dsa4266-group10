"""One file format for every model's results, so charts never depend on model code.

Each run writes to ``reports/results/<model>/seed<seed>/``:

- ``scores.csv``: one row per scored minute; columns ``timestamp, part, score``.
  ``part`` is a ``load_dataset`` key (``"fit"``, ``"tune"``, ``"calib"`` or
  ``"test"``). For windowed models ``timestamp`` is the window's final minute.
- ``meta.json``: at least ``model``, ``seed`` and ``threshold``, plus anything
  else worth reporting (settings, training seconds, parameter count).
- ``latents.npy`` (optional): row i is the embedding of scores row i.
- ``feature_errors.npy`` (optional): row i is the per-feature squared error of
  scores row i, for "which features caused this alert".

Charts and tables read only these files. That lets the charts be built on PCA
in week 2 and rerun unchanged on every other model.
"""

from __future__ import annotations

import json
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
    return root / model / f"seed{seed}"


def _json_default(obj):
    """float32 thresholds and int64 seeds come from NumPy, and json.dump rejects them."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_run(
    directory: Path,
    *,
    scores: pd.DataFrame,
    meta: dict,
    latents: np.ndarray | None = None,
    feature_errors: np.ndarray | None = None,
) -> None:
    """Write a run in the format above, creating ``directory`` if needed.

    NumPy scalar values in ``meta`` (a float32 threshold, an int64 seed) are
    converted to plain Python types with ``.item()`` before the JSON is written,
    because ``json.dump`` rejects them.

    Raises:
        ValueError: if ``scores`` lacks a ``SCORE_COLUMNS`` column, ``meta``
            lacks a ``REQUIRED_META`` key, or an optional array's row count
            differs from ``len(scores)``. Bad files caught here would otherwise
            surface as a wrong chart two weeks later.
    """
    # Validate everything first so a bad call never leaves a half-written run.
    missing_columns = [c for c in SCORE_COLUMNS if c not in scores.columns]
    if missing_columns:
        raise ValueError(f"scores is missing columns: {missing_columns}")
    missing_meta = [k for k in REQUIRED_META if k not in meta]
    if missing_meta:
        raise ValueError(f"meta is missing keys: {missing_meta}")
    optional = {"latents": latents, "feature_errors": feature_errors}
    for name, array in optional.items():
        if array is not None and len(array) != len(scores):
            raise ValueError(f"{name} has {len(array)} rows but scores has {len(scores)}")

    # Serialise meta before touching any file: if it holds something JSON cannot
    # store, this raises here and nothing has been written yet.
    meta_text = json.dumps(meta, indent=2, default=_json_default)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # float_format keeps 9 significant digits, enough to round-trip float32.
    # The default CSV text is shorter, so a score sitting exactly on the
    # threshold could flip from "no alert" to "alert" after load_run.
    scores.to_csv(directory / "scores.csv", index=False, float_format="%.9g")
    (directory / "meta.json").write_text(meta_text)

    for name, array in optional.items():
        path = directory / f"{name}.npy"
        if array is not None:
            np.save(path, np.asarray(array))
        elif path.exists():
            path.unlink()  # don't leave a stale file from an earlier run


def load_run(directory: Path) -> Run:
    """Read a run written by ``save_run``; ``timestamp`` comes back as datetime."""
    directory = Path(directory)
    scores = pd.read_csv(directory / "scores.csv", parse_dates=["timestamp"])
    with open(directory / "meta.json") as f:
        meta = json.load(f)

    def _load_optional(name: str) -> np.ndarray | None:
        path = directory / f"{name}.npy"
        return np.load(path) if path.exists() else None

    return Run(
        scores=scores,
        meta=meta,
        latents=_load_optional("latents"),
        feature_errors=_load_optional("feature_errors"),
    )
