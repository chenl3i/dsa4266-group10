"""The linear baseline: reconstruct each minute from its top principal components.

Every autoencoder has to beat this to justify its extra complexity. Its score
matches ``train.score_autoencoder`` exactly, so it plugs into the same
evaluation and charts.
"""

from __future__ import annotations

import numpy as np


class PCAScorer:
    """Wraps ``sklearn.decomposition.PCA``.

    Attributes:
        explained_variance_ratio_: share of fit-part variance per component.
            Used to choose ``n_components``. Set by ``fit``.
    """

    explained_variance_ratio_: np.ndarray

    def __init__(self, n_components: int) -> None:
        raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")

    def fit(self, X: np.ndarray) -> PCAScorer:
        """Fit on the fit part's preprocessed rows. Returns self."""
        raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(scores, feature_errors)``: squared error of ``inverse_transform(transform(X))``
        per feature, and its mean per row."""
        raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Component coordinates ``(rows, n_components)``, for the latent-space plots."""
        raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")
