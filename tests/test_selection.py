import numpy as np
import pytest

from ugr16ad.evaluate import fit_threshold
from ugr16ad.paths import RESULTS_DIR
from ugr16ad.selection import check_blind_root, selection_score, tune_sanity


def test_tune_sanity_catches_large_spikes():
    X = np.random.default_rng(0).normal(size=(400, 6))

    def score_rows(Z):
        # A perfect detector of the spike: its score is the row's largest value.
        return Z.max(axis=1), np.arange(len(Z))

    clean_threshold = fit_threshold(score_rows(X)[0])
    result = tune_sanity(X, score_rows, clean_threshold, seeds=(0, 1))
    big = {(c["magnitude"], c["n_features"]): c for c in result}[(4.0, 5)]
    assert big["recall"] > 0.5
    assert big["auc"] > 0.9


def test_selection_score_is_mean_auc_plus_largest_cell_ap():
    cells = [
        {"magnitude": 2.0, "n_features": 1, "auc": 0.5, "ap": 0.9, "recall": 0.0},
        {"magnitude": 4.0, "n_features": 5, "auc": 1.0, "ap": 0.4, "recall": 0.0},
    ]
    assert selection_score({"tune_sanity": cells}) == pytest.approx(0.75 + 0.4)


def test_check_blind_root(tmp_path):
    with pytest.raises(ValueError, match="blind"):
        check_blind_root(RESULTS_DIR / "x")
    check_blind_root(tmp_path)
