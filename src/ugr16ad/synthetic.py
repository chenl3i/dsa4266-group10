"""Fake anomalies for the validation sanity check.

Validation has no labelled attacks, so this is the only way to check before
test that a model reacts to anything at all. Inputs are already standardised
with fit-part statistics, so a shift of ``magnitude`` means "this many fit-period
standard deviations" and uses no information from later data.
"""

from __future__ import annotations

import numpy as np


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


SANITY_RATE = 0.01
SANITY_SEED = 0
SANITY_GRID = ((2, 1), (2, 5), (4, 1), (4, 5))


def sanity_check(X, score_rows, *, rate=SANITY_RATE, seed=SANITY_SEED):
    """ROC-AUC on altered calibration copies, aligned to scored endpoint rows.

    Inject before window construction. The callback returns (scores, positions).
    No thresholds or model settings are changed by this diagnostic.
    """
    from sklearn.metrics import roc_auc_score

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
