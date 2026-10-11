"""Turn raw counts into model inputs: log(1+x), drop constant columns, standardise.

Everything here is learned from the fit part only and then applied unchanged to
tune, calib and test. Fitting on later data would let the scaler peek at the
traffic the models are meant to be judged on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Preprocessor:
    """log1p -> drop columns constant in the fit part -> standard-scale.

    Mostly-zero columns are kept on purpose: a rarely used port can be exactly
    what an attack lights up. Standard scaling uses the population std
    (``ddof=0``), as scikit-learn's ``StandardScaler`` does.

    Attributes:
        feature_names_: the kept column names, in input order. Set by ``fit``.
    """

    feature_names_: list[str]

    def fit(self, X: pd.DataFrame) -> Preprocessor:
        """Learn the kept columns, and their log1p means and stds, from ``X``. Returns self."""
        raise NotImplementedError("Kesha, week 1: see docs/team-plan.md")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted steps. Returns float32 ``(rows, len(feature_names_))``.

        Raises:
            RuntimeError: if called before ``fit``.
            ValueError: if ``X`` lacks any column in ``feature_names_``. Columns
                dropped as constant at fit time may be absent.
        """
        raise NotImplementedError("Kesha, week 1: see docs/team-plan.md")
