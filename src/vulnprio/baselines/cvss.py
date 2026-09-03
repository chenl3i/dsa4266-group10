"""Rank by CVSS base score - the strategy the project is testing against.

Learns nothing, so ``fit`` is a no-op. Its one subtlety is severe and easy to
miss: **CVSS scores are massively tied.** The score is a discrete formula, so
tens of thousands of CVEs share a base score of exactly 9.8, and 9.8 is above
the cut for most realistic patch budgets. That means the "top 10% by CVSS" is
not well defined - it depends entirely on how ties are broken.

Reported precision@k for this baseline can swing by a factor of two purely on
tie policy, which is why tie handling is an explicit, documented decision in
:func:`vulnprio.evaluation.metrics.order_by_score` rather than an accident of
sort order. Getting this right is what makes the comparison honest; a
convenient tie order is the easiest way to accidentally beat the baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vulnprio.baselines.registry import register_baseline
from vulnprio.data import schema
from vulnprio.models.base import Ranker


@register_baseline
class CVSSBaseline(Ranker):
    name = "cvss"
    produces_probabilities = False

    def __init__(self, score_column: str = schema.CVSS_BASE_SCORE) -> None:
        self.score_column = score_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CVSSBaseline:
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self.score_column not in X.columns:
            raise KeyError(
                f"CVSS baseline needs column {self.score_column!r}. Present: {list(X.columns)[:10]}"
            )
        # Unscored CVEs sort last rather than raising: they exist in NVD, and
        # dropping them would quietly change the denominator of precision@k.
        return pd.to_numeric(X[self.score_column], errors="coerce").fillna(-1.0).to_numpy()

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "score_column": self.score_column}
