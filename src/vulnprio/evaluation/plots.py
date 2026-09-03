"""Figures for the report.

Four that earn their place, and one that does not:

``precision_recall_curve``
    The headline. One line per ranker. Under this much imbalance it is far more
    readable than an ROC curve, where every method hugs the top-left corner.

``precision_at_k_curve``
    precision@k across the full sweep of budgets. This is the figure that
    directly answers "if we could patch k%, how much better off are we?" and is
    usually the most persuasive single image in the report.

``gain_curve``
    Cumulative fraction of exploited CVEs captured as budget increases, against
    the random diagonal. The security-team-legible version of the same story.

``calibration_curve``
    Predicted probability against observed frequency, for rankers where
    ``produces_probabilities`` is True. Ranking quality and calibration are
    different things, and a well-ranked but badly calibrated model is still
    useful - worth showing that you know the difference.

Not included: a ROC curve as the lead figure. Report the AUC number for
comparability, but leading with the curve misrepresents performance here.

Every function takes ``y_true`` plus the ranker results, so nothing is refitted
to draw a picture.

Implemented in plan Task 17.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from vulnprio.evaluation.results import RankerResult


def precision_recall_curve(y_true: np.ndarray, results: dict[str, RankerResult]) -> Figure:
    """One PR curve per ranker, with the base rate marked."""
    raise NotImplementedError("Implemented in plan Task 17 (figures).")


def precision_at_k_curve(results: dict[str, RankerResult]) -> Figure:
    """precision@k against patch budget, one line per ranker.

    Reads the budgets straight off each result's table, so it always plots
    exactly the budgets that were evaluated.
    """
    raise NotImplementedError("Implemented in plan Task 17 (figures).")


def gain_curve(y_true: np.ndarray, results: dict[str, RankerResult]) -> Figure:
    """Cumulative exploited-CVE capture against budget, with the random diagonal."""
    raise NotImplementedError("Implemented in plan Task 17 (figures).")


def calibration_curve(
    y_true: np.ndarray, results: dict[str, RankerResult], n_bins: int = 10
) -> Figure:
    """Predicted probability against observed frequency, for probabilistic rankers."""
    raise NotImplementedError("Implemented in plan Task 17 (figures).")


def save(figure: Figure, path: Path, *, dpi: int = 200) -> Path:
    """Write a figure to ``reports/figures/``, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
