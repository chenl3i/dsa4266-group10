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
        logged = np.log1p(X.to_numpy(dtype=np.float64))
        scales = logged.std(axis=0)
        keep = scales > 0

        self.feature_names_ = X.columns[keep].tolist()
        self.mean_ = logged.mean(axis=0)[keep]
        self.scale_ = scales[keep]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted steps. Returns float32 ``(rows, len(feature_names_))``.

        Raises:
            RuntimeError: if called before ``fit``.
            ValueError: if ``X`` lacks any column in ``feature_names_``. Columns
                dropped as constant at fit time may be absent.
        """
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("Preprocessor must be fitted before transform")

        missing = [name for name in self.feature_names_ if name not in X.columns]
        if missing:
            raise ValueError(f"X is missing fitted feature columns: {missing}")

        logged = np.log1p(X.loc[:, self.feature_names_].to_numpy(dtype=np.float64))
        return ((logged - self.mean_) / self.scale_).astype(np.float32)
