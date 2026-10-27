"""Fake anomalies for the validation sanity check.

Validation has no labelled attacks, so this is the only way to check before
test that a model reacts to anything at all. Inputs are already standardised
with fit-part statistics, so a shift of ``magnitude`` means "this many fit-period
standard deviations" and uses no information from later data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

# Settled by the team on 26 Oct. The grid spans small to large spikes so the check
# shows where each model starts reacting, instead of every model scoring AUC = 1.
SANITY_RATE = 0.01
SANITY_GRID = ((2.0, 1), (2.0, 5), (4.0, 1), (4.0, 5))  # (magnitude, n_features)
# Fixed, not the run seed: every model and seed sees the same altered rows, so
# their sanity AUCs are comparable.
SANITY_SEED = 0


def inject_anomalies(
    X: np.ndarray, *, rate: float, magnitude: float, n_features: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(copy of X with anomalies, bool mask of altered rows)``.

    Picks ``round(rate * len(X))`` rows at random. In each, adds ``magnitude``
    to ``n_features`` randomly chosen columns (different columns per row).
    ``X`` is 2-D ``(rows, features)`` and is never modified in place. The same
    ``seed`` gives the same result.
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols = X.shape
    rows = rng.choice(n_rows, size=round(rate * n_rows), replace=False)
    # Sorting a random key per row gives each row its own distinct columns.
    cols = np.argsort(rng.random((len(rows), n_cols)), axis=1)[:, :n_features]

    out = X.copy()
    out[rows[:, None], cols] += magnitude
    mask = np.zeros(n_rows, dtype=bool)
    mask[rows] = True
    return out, mask


def sanity_check(
    X: np.ndarray,
    score_rows: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    rate: float = SANITY_RATE,
    seed: int = SANITY_SEED,
) -> list[dict[str, float]]:
    """AUC of a model's scores on rows injected into ``X``, one dict per grid entry.

    ``X`` is preprocessed calib. ``score_rows(X_injected)`` returns ``(scores,
    positions)``, the row each score belongs to: windowed models skip the first
    minutes of each segment, so only scored rows can count. An injected minute
    also raises the scores of later windows that contain it; those windows stay
    labelled clean, because the score is the final minute's. AUC is ``nan`` when
    the scored rows hold one class (too few rows for ``rate`` to pick any).
    """
    result = []
    for magnitude, n_features in SANITY_GRID:
        injected, mask = inject_anomalies(
            X, rate=rate, magnitude=magnitude, n_features=n_features, seed=seed
        )
        scores, positions = score_rows(injected)
        y = mask[positions]
        auc = float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else float("nan")
        result.append({"magnitude": magnitude, "n_features": n_features, "auc": auc})
    return result
