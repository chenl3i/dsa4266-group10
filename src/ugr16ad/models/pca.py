"""The linear baseline: reconstruct each minute from its top principal components.

Every autoencoder has to beat this to justify its extra complexity. Its score
matches ``train.score_autoencoder`` exactly, so it plugs into the same
evaluation and charts.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class PCAScorer:
    """Wraps ``sklearn.decomposition.PCA``.

    Attributes:
        explained_variance_ratio_: share of fit-part variance per component.
            Used to choose ``n_components``. Set by ``fit``.
    """

    explained_variance_ratio_: np.ndarray

    def __init__(self, n_components: int) -> None:
        if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
            raise ValueError("n_components must be a positive integer")
        if n_components < 1:
            raise ValueError("n_components must be a positive integer")
        self.pca = PCA(n_components=n_components, svd_solver="full", whiten=False)

    def fit(self, X: np.ndarray) -> PCAScorer: #Fit on the fit part's preprocessed rows. Returns self
        self.pca.fit(X)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_.copy()
        return self

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: #Return ``(scores, feature_errors)`` for each row of ``X``
        """Return ``(scores, feature_errors)`` for each row of ``X``.

        ``feature_errors`` is the squared error of ``inverse_transform(transform(X))``
        per feature, shape ``(rows, features)``; ``scores`` is its per-row mean.
        """
        reconstruction = self.pca.inverse_transform(self.transform(X))
        feature_errors = (np.asarray(X) - reconstruction) ** 2
        return feature_errors.mean(axis=1), feature_errors

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Component coordinates ``(rows, n_components)``, for the latent-space plots.

        A plain centred projection onto the components (not whitened).
        """
        return self.pca.transform(X)
