import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ugr16ad.data import ATTACK_LABELS, Part
from ugr16ad.evaluate import fit_threshold
from ugr16ad.results import load_run
from ugr16ad.synthetic import SANITY_GRID
from ugr16ad.windows import eligible_positions

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_cnn.py"
_spec = importlib.util.spec_from_file_location("run_cnn", SCRIPT)
run_cnn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_cnn)

WINDOW = 5
SETTINGS = dict(window_length=WINDOW, epochs=2, hidden_channels=4, latent_dim=2)


def _part(rng, start, segment, attack_rows=()):
    n = len(segment)
    index = pd.date_range(start, periods=n, freq="min", name="timestamp")
    # Poisson counts vary in every part, so Preprocessor keeps all 3 columns.
    X = pd.DataFrame(rng.poisson(5, size=(n, 3)), index=index, columns=["a", "b", "c"])
    labels = pd.DataFrame(0, index=index, columns=list(ATTACK_LABELS))
    labels.iloc[list(attack_rows), 0] = 1
    return Part(
        X=X,
        y=labels.sum(axis=1) > 0,
        labels=labels,
        segment=pd.Series(segment, index=index),
    )


def tiny_parts():
    rng = np.random.default_rng(0)
    return {
        "fit": _part(rng, "2016-01-01", [0] * 30 + [1] * 30),
        "tune": _part(rng, "2016-02-01", [0] * 30),
        "calib": _part(rng, "2016-03-01", [0] * 30),
        "test": _part(rng, "2016-04-01", [0] * 30, attack_rows=range(10, 15)),
    }


@pytest.fixture(scope="module")
def parts():
    return tiny_parts()


@pytest.fixture(scope="module")
def run_path(parts, tmp_path_factory):
    return run_cnn.run(parts, seed=0, out_root=tmp_path_factory.mktemp("out"), **SETTINGS)


def test_writes_a_loadable_cnn_run(run_path):
    run = load_run(run_path)
    assert run.meta["model"] == "cnn"
    assert run_path.parts[-2:] == ("cnn", "seed0")
    assert len(pd.read_csv(run_path / "history.csv")) == run.meta["epochs_run"]


def test_meta_has_the_comparison_table_keys(run_path):
    meta = load_run(run_path).meta
    for key in ("threshold", "n_parameters", "train_seconds", "inference_seconds"):
        assert meta[key] is not None


def test_scores_the_shared_eligible_positions(parts, run_path):
    scores = load_run(run_path).scores
    for name, part in parts.items():
        expected = part.X.index[eligible_positions(part.segment.to_numpy(), WINDOW)]
        got = scores.loc[scores["part"] == name, "timestamp"]
        assert list(got) == list(expected)


def test_threshold_is_the_calib_quantile(run_path):
    run = load_run(run_path)
    calib = run.scores.loc[run.scores["part"] == "calib", "score"].to_numpy()
    # scores.csv keeps 9 significant digits, so the reloaded quantile matches to ~1e-8.
    assert run.meta["threshold"] == pytest.approx(fit_threshold(calib), rel=1e-6)


def test_meta_records_sanity_check(run_path):
    checks = load_run(run_path).meta["sanity_check"]
    assert [(c["magnitude"], c["n_features"]) for c in checks] == [tuple(g) for g in SANITY_GRID]
