"""Turn anomaly scores into the numbers the report quotes.

Two kinds of result are kept apart on purpose. Metrics at the frozen threshold
are what an analyst would actually have seen. Ranking metrics sweep every
threshold on test, so they are retrospective comparisons only and never replace
the frozen threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _require_finite(scores: np.ndarray, name: str = "scores") -> None:
    """A NaN score would silently never alert, or make the threshold NaN, so fail loudly."""
    if not np.isfinite(np.asarray(scores, dtype=float)).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def fit_threshold(calib_scores: np.ndarray, quantile: float = 0.99) -> float:
    """The alarm threshold: the ``quantile`` of calibration scores (``np.quantile``).

    At 0.99 about 1% of calibration minutes alert. The threshold is then frozen
    and applied unchanged to test.

    Returns a plain Python ``float``, not ``np.float32``: scores are float32 in
    the real flow, and ``json.dump`` rejects NumPy scalars when the threshold
    goes into ``meta.json``.
    """
    _require_finite(calib_scores, "calib_scores")
    return float(np.quantile(calib_scores, quantile))


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion counts and rates at one threshold. A minute alerts when ``score > threshold``.

    Keys: ``tp, fp, fn, tn, recall, precision, f1, fpr, alert_rate``.
    ``alert_rate`` is the share of all minutes that alert; it is the analyst's
    workload. With no alerts, precision and F1 are 0.0 rather than undefined,
    so a silent model scores as useless instead of crashing the table.
    """
    _require_finite(scores)
    y = np.asarray(y_true).astype(bool)
    alerts = np.asarray(scores) > threshold

    tp = int(np.sum(alerts & y))
    fp = int(np.sum(alerts & ~y))
    fn = int(np.sum(~alerts & y))
    tn = int(np.sum(~alerts & ~y))

    nan = float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else nan  # undefined without positives
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # no alerts -> 0.0
    f1_denominator = 2 * tp + fp + fn  # F1 = 2PR / (P + R) = 2tp / (2tp + fp + fn)
    f1 = 2 * tp / f1_denominator if f1_denominator > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else nan  # undefined without negatives
    alert_rate = (tp + fp) / len(y) if len(y) > 0 else nan

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": fpr,
        "alert_rate": alert_rate,
    }


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
    _require_finite(scores)
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores)

    # With only one class present, AP and ROC curves are undefined.
    if y.all() or not y.any():
        nan = float("nan")
        return {"average_precision": nan, "recall_at_max_fpr": nan, "partial_auc": nan}

    # drop_intermediate=False keeps every threshold, so no point inside the FPR
    # budget is skipped when we take the best recall.
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    return {
        "average_precision": float(average_precision_score(y, s)),
        "recall_at_max_fpr": float(tpr[fpr <= max_fpr].max()),
        "partial_auc": float(roc_auc_score(y, s, max_fpr=max_fpr)),
    }


def per_category_recall(labels: pd.DataFrame, alerts: np.ndarray) -> pd.DataFrame:
    """Recall per attack type, counted in minutes, not episodes.

    ``labels`` is ``Part.labels`` (one count column per category); a minute is
    positive for a category when its count is > 0. Returns one row per column of
    ``labels`` (index = category) with ``n_positive``, ``n_detected`` and
    ``recall``. ``recall`` is NaN when ``n_positive`` is 0, because SSH scan and
    spam never occur in test and "0% detected" would be a false claim.
    """
    alerts = np.asarray(alerts).astype(bool)
    if len(alerts) != len(labels):
        raise ValueError(
            f"alerts has {len(alerts)} rows but labels has {len(labels)}; "
            "windowed models score fewer minutes, so align labels to the scored timestamps first"
        )

    rows = {}
    for category in labels.columns:
        positive = labels[category].to_numpy() > 0
        n_positive = int(positive.sum())
        n_detected = int((positive & alerts).sum())
        rows[category] = {
            "n_positive": n_positive,
            "n_detected": n_detected,
            "recall": n_detected / n_positive if n_positive > 0 else float("nan"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
