import numpy as np

from stubs import todo
from ugr16ad.models.pca import PCAScorer


def _low_rank(n=100):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, 2)) @ rng.normal(size=(2, 5))


@todo
def test_pca_reconstructs_its_own_subspace():
    X = _low_rank()
    scores, errors = PCAScorer(2).fit(X).score(X)
    assert scores.shape == (100,)
    assert errors.shape == (100, 5)
    np.testing.assert_allclose(scores, 0, atol=1e-8)


@todo
def test_point_off_the_subspace_scores_higher():
    X = _low_rank()
    pca = PCAScorer(2).fit(X)
    off = X[:1] + np.array([[5.0, -5.0, 5.0, -5.0, 5.0]])
    assert pca.score(off)[0][0] > pca.score(X)[0].max()


@todo
def test_score_is_mean_of_feature_errors():
    X = _low_rank()
    noisy = X + np.random.default_rng(1).normal(scale=0.1, size=X.shape)
    scores, errors = PCAScorer(1).fit(X).score(noisy)
    np.testing.assert_allclose(scores, errors.mean(axis=1))


@todo
def test_transform_and_explained_variance():
    pca = PCAScorer(2).fit(_low_rank())
    assert pca.transform(_low_rank()).shape == (100, 2)
    assert pca.explained_variance_ratio_.shape == (2,)
    np.testing.assert_allclose(pca.explained_variance_ratio_.sum(), 1.0, atol=1e-8)


@todo
def test_score_matches_a_manual_projection_with_the_fit_mean():
    # Scoring rows other than the fit rows must centre on the *fit* mean, not on
    # the mean of the rows being scored.
    X = _low_rank()
    other = _low_rank() * 0.5 + 3.0 + np.random.default_rng(2).normal(size=(100, 5))
    scores, errors = PCAScorer(2).fit(X).score(other)
    mean = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    basis = vt[:2]
    recon = (other - mean) @ basis.T @ basis + mean
    np.testing.assert_allclose(errors, (other - recon) ** 2, atol=1e-8)
    np.testing.assert_allclose(scores, errors.mean(axis=1))


@todo
def test_fit_returns_self_and_transform_is_the_centred_projection():
    X = _low_rank()
    pca = PCAScorer(2)
    assert pca.fit(X) is pca
    mean = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    # Component signs are arbitrary, so compare magnitudes.
    np.testing.assert_allclose(np.abs(pca.transform(X)), np.abs((X - mean) @ vt[:2].T), atol=1e-8)


@todo
def test_explained_variance_ratio_is_sorted_and_partial_when_components_are_dropped():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)) * np.array([5.0, 3.0, 2.0, 1.0, 0.5])
    ratio = PCAScorer(3).fit(X).explained_variance_ratio_
    assert np.all(np.diff(ratio) <= 0)
    assert 0.9 < ratio.sum() < 1.0
