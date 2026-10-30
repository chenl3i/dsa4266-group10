import json

import numpy as np
import pandas as pd
import pytest

from ugr16ad import experiments
from ugr16ad.data import ATTACK_LABELS
from ugr16ad.results import load_run, run_dir


def test_elbow_uses_curve_not_just_lowest_loss():
    assert experiments.elbow_index([4, 8, 16], [1.0, 0.5, 0.45]) == 1
    assert experiments.elbow_index([4, 8, 16], [1.0, 1.1, 1.2]) == 0


def test_tune_then_frozen_evaluation(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    calls = []

    def read_split(split):
        calls.append(split)
        n = 100 if split == "train" else 30
        index = pd.date_range(
            "2016-03-01" if split == "train" else "2016-07-01",
            periods=n,
            freq="min",
            name="timestamp",
        )
        X = pd.DataFrame(rng.lognormal(size=(n, 12)), index=index)
        Y = pd.DataFrame(0, index=index, columns=list(ATTACK_LABELS))
        if split == "test":
            Y.iloc[10:15, 0] = 1
        return X, Y

    # Fixed frames ensure subsequent reads are exactly the same data.
    frames = {split: read_split(split) for split in ("train", "test")}
    calls.clear()

    def read_fixed(split):
        calls.append(split)
        return frames[split]

    monkeypatch.setattr(experiments, "read_split", read_fixed)
    monkeypatch.setattr(experiments, "FIT_ROWS", 60)
    monkeypatch.setattr(experiments, "UGR16_DIR", tmp_path)
    monkeypatch.setattr(
        experiments, "TRAINING", dict(epochs=2, batch_size=20, lr=0.001, patience=2, device="cpu")
    )
    root = tmp_path / "run"
    experiments.tune_run(root, 0.5)
    assert calls == ["train"]  # No test access during selection.
    audit = json.loads((root / "reproducibility.json").read_text())
    assert audit["weights_identical"] and audit["scores_identical"]
    frozen = (root / "selected.json").read_text()
    shared = tmp_path / "shared-results"
    monkeypatch.setattr(experiments, "RESULTS_DIR", shared)
    experiments.evaluate_run(root, window=3)
    assert (root / "selected.json").read_text() == frozen
    directory = run_dir("mlp", 0, root=shared)
    run = load_run(directory)
    assert (directory / run.meta["manifest"]).resolve() == (root / "manifest.json").resolve()
    assert (root / "evaluation-window3" / "comparison.csv").is_file()
    assert not (root / "evaluation-window3" / "mlp").exists()
    for model, seeds in (("pca", [0]), ("mlp", [0, 1, 2])):
        assert all((run_dir(model, seed, root=shared) / "scores.csv").exists() for seed in seeds)
    before = (directory / "scores.csv").read_bytes()
    with pytest.raises(FileExistsError, match="Run already exists"):
        experiments.evaluate_run(root, window=4)
    assert (directory / "scores.csv").read_bytes() == before
    assert not (root / "evaluation-window4").exists()
    calib = run.scores[run.scores.part == "calib"].score
    assert len(calib) == 18
    assert np.isclose(run.meta["threshold"], np.quantile(calib, 0.99))
    assert len(run.scores[run.scores.part == "test"]) == 28
    assert run.feature_errors.shape == (64, 12)
    assert sum(run.meta["metrics"][k] for k in ("tp", "fp", "tn", "fn")) == 28


def test_sanity_grid_uses_full_calibration_copy_and_endpoint_labels():
    from sklearn.metrics import roc_auc_score

    from ugr16ad.synthetic import SANITY_GRID, inject_anomalies, sanity_check

    X = np.zeros((1000, 6), dtype=np.float32)
    positions = np.arange(19, len(X))
    seen = []

    def score_rows(values):
        seen.append(values.copy())
        return (values[positions] ** 2).mean(axis=1), positions

    result = sanity_check(X, score_rows)
    assert len(result) == 4
    np.testing.assert_array_equal(X, np.zeros_like(X))
    for row, observed, (magnitude, features) in zip(result, seen, SANITY_GRID, strict=True):
        expected, mask = inject_anomalies(
            X, rate=0.01, magnitude=magnitude, n_features=features, seed=0
        )
        np.testing.assert_array_equal(observed, expected)
        assert mask.sum() == 10
        assert row["auc"] == roc_auc_score(mask[positions], (expected[positions] ** 2).mean(axis=1))
