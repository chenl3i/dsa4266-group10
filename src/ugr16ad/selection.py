"""Config selection shared by every model's run script and sweep.

One rule, used for every model, so configs are chosen the same way everywhere and
on tune only: test labels never pick settings and calib only sets the threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ugr16ad.paths import RESULTS_DIR
from ugr16ad.synthetic import SANITY_GRID, SANITY_RATE, inject_anomalies

# Several injection draws, because one draw puts only ~98 anomalies in each tune
# grid cell: too few to rank configs that differ by a few points of recall.
TUNE_SANITY_SEEDS = (0, 1, 2, 3, 4)


def tune_sanity(
    X: np.ndarray,
    score_rows,
    threshold: float,
    *,
    seeds: tuple[int, ...] = TUNE_SANITY_SEEDS,
) -> list[dict[str, float]]:
    """Synthetic-anomaly detection on tune, per ``SANITY_GRID`` cell, for picking a config.

    Test labels may not choose settings and calib only sets the threshold, so the
    sweep ranks configs on anomalies injected into tune. ``threshold`` is the 99th
    percentile of clean tune scores, so ``recall`` mimics recall at FPR <= 1%.
    AUC, AP and recall are averaged over ``seeds``; ``nan`` when the scored rows
    hold one class. ``score_rows`` follows ``synthetic.sanity_check``.
    """
    result = []
    for magnitude, n_features in SANITY_GRID:
        per_seed = []
        for seed in seeds:
            injected, mask = inject_anomalies(
                X, rate=SANITY_RATE, magnitude=magnitude, n_features=n_features, seed=seed
            )
            scores, positions = score_rows(injected)
            y = mask[positions]
            if 0 < y.sum() < len(y):
                per_seed.append(
                    (
                        roc_auc_score(y, scores),
                        average_precision_score(y, scores),
                        (scores[y] > threshold).mean(),
                    )
                )
        auc, ap, recall = np.mean(per_seed, axis=0) if per_seed else (np.nan,) * 3
        result.append(
            {
                "magnitude": magnitude,
                "n_features": n_features,
                "auc": float(auc),
                "ap": float(ap),
                "recall": float(recall),
            }
        )
    return result


def selection_score(meta: dict) -> float:
    """One number per run that ranks configs; higher is better.

    Inputs available: ``meta["tune_sanity"]``, one dict per ``SANITY_GRID`` cell
    with ``magnitude``, ``n_features``, ``auc``, ``ap`` and ``recall`` (at the 99th
    percentile of clean tune scores); and ``meta["best_tune_loss"]``.

    Fixed before stage 1 and never changed after seeing results.

    Mean AUC over all four cells, so the subtle spikes count, plus AP on the
    largest cell (4 sigma x 5 features), the only cell where AP and recall rise
    above the 1% base rate, standing in for the AP and recall@FPR<=1% targets.
    ``best_tune_loss`` is left out: a lower loss can mean the model also
    reconstructs anomalies well, which hides them.
    """
    cells = meta["tune_sanity"]
    mean_auc = float(np.mean([c["auc"] for c in cells]))
    largest = max(cells, key=lambda c: (c["magnitude"], c["n_features"]))
    return mean_auc + largest["ap"]


def check_blind_root(out_root: Path) -> None:
    """Refuse a blind run that would land in ``RESULTS_DIR``.

    The comparison table reads every run there as a finished one, so a blind run
    (no test scores) would be mistaken for a real result.
    """
    if Path(out_root).resolve().is_relative_to(RESULTS_DIR.resolve()):
        raise ValueError(f"blind runs must not be written under {RESULTS_DIR}")
