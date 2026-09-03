"""Rank by EPSS score - the published state of the art for this task.

FIRST's Exploit Prediction Scoring System estimates the probability that a CVE
will be exploited within the next 30 days. It is what a well-resourced security
team would actually use today, which makes it the honest comparison point.

Two things the report must state plainly rather than bury:

1. **EPSS is trained partly on KEV.** Evaluating it against a KEV-derived target
   is somewhat favourable to EPSS. That is a limitation of the comparison, not a
   reason to omit it - and the direction of the bias is worth naming, because it
   means a model that *matches* EPSS here is doing better than it looks.

2. **The score must be taken as of a valid date.** EPSS publishes daily. Using
   today's score to rank a 2023 test set is straightforward temporal leakage:
   today's score already reflects whether the CVE was exploited. The
   ``as_of_date`` argument is required, not optional, so the mistake cannot be
   made by forgetting a parameter.

Implemented in plan Task 14.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from vulnprio.baselines.registry import register_baseline
from vulnprio.models.base import Ranker

EPSS_SCORE_COLUMN = "epss_score"


@register_baseline
class EPSSBaseline(Ranker):
    name = "epss"
    produces_probabilities = True

    def __init__(self, as_of_date: dt.date, score_column: str = EPSS_SCORE_COLUMN) -> None:
        self.as_of_date = as_of_date
        self.score_column = score_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> EPSSBaseline:
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in plan Task 14 (EPSS baseline).")

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "as_of_date": self.as_of_date.isoformat(),
            "score_column": self.score_column,
        }
