"""Turn anomaly scores into the numbers the report quotes.

Two kinds of result are kept apart on purpose. Metrics at the frozen threshold
are what an analyst would actually have seen. Ranking metrics sweep every
threshold on test, so they are retrospective comparisons only and never replace
the frozen threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_threshold(calib_scores: np.ndarray, quantile: float = 0.99) -> float:
    """The alarm threshold: the ``quantile`` of calibration scores (``np.quantile``).

    At 0.99 about 1% of calibration minutes alert. The threshold is then frozen
    and applied unchanged to test.

    Returns a plain Python ``float``, not ``np.float32``: scores are float32 in
    the real flow, and ``json.dump`` rejects NumPy scalars when the threshold
    goes into ``meta.json``.
    """
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion counts and rates at one threshold. A minute alerts when ``score > threshold``.

    Keys: ``tp, fp, fn, tn, recall, precision, f1, fpr, alert_rate``.
    ``alert_rate`` is the share of all minutes that alert; it is the analyst's
    workload. With no alerts, precision and F1 are 0.0 rather than undefined,
    so a silent model scores as useless instead of crashing the table.
    """
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")


def ranking_metrics(
    y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01
) -> dict[str, float]:
    """Threshold-free comparisons on test.

    Keys:
        average_precision: ``sklearn.metrics.average_precision_score``.
        recall_at_max_fpr: best recall over thresholds whose FPR <= ``max_fpr``
            (from ``sklearn.metrics.roc_curve``).
        partial_auc: ``roc_auc_score(..., max_fpr=max_fpr)``, the standardised
            (McClish) partial AUC over FPR 0..``max_fpr``.
    """
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")


def per_category_recall(labels: pd.DataFrame, alerts: np.ndarray) -> pd.DataFrame:
    """Recall per attack type, counted in minutes, not episodes.

    ``labels`` is ``Part.labels`` (one count column per category); a minute is
    positive for a category when its count is > 0. Returns one row per column of
    ``labels`` (index = category) with ``n_positive``, ``n_detected`` and
    ``recall``. ``recall`` is NaN when ``n_positive`` is 0, because SSH scan and
    spam never occur in test and "0% detected" would be a false claim.
    """
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")
