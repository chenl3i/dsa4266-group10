import numpy as np

from ugr16ad.synthetic import SANITY_GRID, inject_anomalies, sanity_check

# Standard-normal rows, like preprocessed calib.
X = np.random.default_rng(1).normal(size=(2000, 8))


def max_abs(X):
    # Oracle scorer: a single-minute model that flags any large value.
    return np.abs(X).max(axis=1), np.arange(len(X))


def test_inject_anomalies_shifts_only_marked_rows():
    # Non-zero base, so "adds magnitude" is distinguishable from "sets to magnitude".
    X = np.ones((1000, 20), dtype=np.float32)
    injected, mask = inject_anomalies(X, rate=0.05, magnitude=4.0, n_features=3, seed=0)
    delta = injected - X
    assert mask.dtype == bool
    assert mask.sum() == 50
    assert np.all(delta[~mask] == 0)
    assert np.all((delta[mask] != 0).sum(axis=1) == 3)
    assert np.allclose(delta[mask][delta[mask] != 0], 4.0)
    # Columns are drawn per row, not once for all rows.
    assert np.unique(np.nonzero(delta[mask])[1]).size > 3
    assert np.all(X == 1), "input must not be modified in place"


def test_inject_anomalies_is_reproducible():
    X = np.zeros((200, 10), dtype=np.float32)
    a, mask_a = inject_anomalies(X, rate=0.1, magnitude=3.0, n_features=2, seed=1)
    b, mask_b = inject_anomalies(X, rate=0.1, magnitude=3.0, n_features=2, seed=1)
    c, _ = inject_anomalies(X, rate=0.1, magnitude=3.0, n_features=2, seed=2)
    assert np.array_equal(a, b) and np.array_equal(mask_a, mask_b)
    assert not np.array_equal(a, c)


def test_sanity_check_follows_the_grid():
    result = sanity_check(X, max_abs)
    assert [(r["magnitude"], r["n_features"]) for r in result] == list(SANITY_GRID)


def test_sanity_check_bigger_anomalies_score_higher():
    by_key = {(r["magnitude"], r["n_features"]): r["auc"] for r in sanity_check(X, max_abs)}
    assert by_key[(4.0, 5)] > 0.95
    assert by_key[(4.0, 5)] > by_key[(2.0, 1)]


def test_sanity_check_uses_only_scored_positions():
    # Drop the first 19 rows, as a 20-minute window scorer would. Scores must be
    # matched to mask[positions]; using the full mask would raise a length error.
    def windowed(X):
        positions = np.arange(19, len(X))
        return np.abs(X[positions]).max(axis=1), positions

    assert all(np.isfinite(r["auc"]) for r in sanity_check(X, windowed))


def test_sanity_check_single_class_is_nan():
    # round(0.01 * 10) = 0 injected rows, so AUC is undefined.
    assert all(np.isnan(r["auc"]) for r in sanity_check(X[:10], max_abs))


def test_sanity_check_is_reproducible_and_leaves_X_alone():
    before = X.copy()
    assert sanity_check(X, max_abs) == sanity_check(X, max_abs)
    np.testing.assert_array_equal(X, before)
