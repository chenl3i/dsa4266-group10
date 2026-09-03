"""Random ranking - the performance floor.

Included because it makes the class imbalance legible inside the results table.
Precision@k under random ordering converges to the base rate, so if the positive
rate is 0.5%, a model scoring 0.05 precision@10% is a 10x lift - a fact that is
much harder to read off an accuracy column.

Seeded from config so the floor is identical across runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vulnprio.baselines.registry import register_baseline
from vulnprio.models.base import Ranker


@register_baseline
class RandomBaseline(Ranker):
    name = "random"
    produces_probabilities = False

    def __init__(self, seed: int = 4266) -> None:
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RandomBaseline:
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(X))

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "seed": self.seed}
