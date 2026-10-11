import numpy as np

from stubs import todo
from ugr16ad.synthetic import inject_anomalies


@todo
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


@todo
def test_inject_anomalies_is_reproducible():
    X = np.zeros((200, 10), dtype=np.float32)
    a, mask_a = inject_anomalies(X, rate=0.1, magnitude=3.0, n_features=2, seed=1)
    b, mask_b = inject_anomalies(X, rate=0.1, magnitude=3.0, n_features=2, seed=1)
    c, _ = inject_anomalies(X, rate=0.1, magnitude=3.0, n_features=2, seed=2)
    assert np.array_equal(a, b) and np.array_equal(mask_a, mask_b)
    assert not np.array_equal(a, c)
