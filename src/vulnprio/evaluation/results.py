"""What one ranker produced on one split.

Carrying the raw scores next to the metric table matters more than it looks.
Bootstrap intervals, PR curves, gain curves and calibration plots all need the
scores, and without this the only options are refitting every ranker per
consumer (slow, and a fresh chance to fit a model on the wrong split) or
threading a parallel dict of scores through every function signature.

One object, computed once, passed around.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RankerResult:
    """Scores and metrics for a single ranker on a single split."""

    #: Registry name of the ranker, e.g. "cvss" or "logistic".
    name: str

    #: One score per row of the scored split, higher = patch sooner.
    scores: np.ndarray

    #: One row per patch budget, from ``evaluate_ranking``.
    table: pd.DataFrame

    #: Whatever the ranker chose to report about itself, from ``Ranker.describe``.
    describe: dict[str, object] = field(default_factory=dict)

    @property
    def headline(self) -> dict[str, float] | None:
        """The bootstrap interval on the primary metric, once computed."""
        return self.table.attrs.get("headline")

    def metric(self, name: str, k_percent: float) -> float:
        """Look up one metric at one budget."""
        row = self.table.loc[self.table["k_percent"] == k_percent]
        if row.empty:
            raise KeyError(
                f"No result at k={k_percent}%. Available: {list(self.table['k_percent'])}"
            )
        return float(row[name].iloc[0])
