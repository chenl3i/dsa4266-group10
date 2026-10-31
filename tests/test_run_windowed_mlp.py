import functools
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ugr16ad.data import ATTACK_LABELS, Part
from ugr16ad.preprocess import Preprocessor
from ugr16ad.results import load_run
from ugr16ad.synthetic import SANITY_GRID
from ugr16ad.train import train_autoencoder

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_windowed_mlp.py"
_spec = importlib.util.spec_from_file_location("run_windowed_mlp", SCRIPT)
run_windowed_mlp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_windowed_mlp)

WINDOW = 5
SETTINGS = dict(hidden_dim=4, dropout=0.1, noise_std=0.1)


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
def mod():
    # Module-level patches, undone when the module's tests finish.
    patch = pytest.MonkeyPatch()
    patch.setattr(run_windowed_mlp, "WINDOW_LENGTH", WINDOW)
    patch.setattr(
        run_windowed_mlp, "train_autoencoder", functools.partial(train_autoencoder, epochs=2)
    )
    yield run_windowed_mlp
    patch.undo()


@pytest.fixture(scope="module")
def inputs(mod):
    parts = tiny_parts()
    prep = Preprocessor().fit(parts["fit"].X)
    features = {name: prep.transform(part.X) for name, part in parts.items()}
    fit = mod.window_part(parts, features, "fit")
    tune = mod.window_part(parts, features, "tune")
    return parts, features, fit, tune


def _run(mod, inputs, out_root, **kwargs):
    parts, features, fit, tune = inputs
    mod.run_seed(parts, features, fit, tune, 0, 2, 1e-3, out_root=out_root, **SETTINGS, **kwargs)
    return out_root / mod.MODEL_NAME / "seed0"


@pytest.fixture(scope="module")
def run_path(mod, inputs, tmp_path_factory):
    return _run(mod, inputs, tmp_path_factory.mktemp("out"))


@pytest.fixture(scope="module")
def blind_path(mod, inputs, tmp_path_factory):
    return _run(mod, inputs, tmp_path_factory.mktemp("blind"), blind=True)


def test_blind_run_never_touches_test(blind_path):
    run = load_run(blind_path)
    assert set(run.scores.part) == {"tune", "calib"}
    for key in ("test_metrics", "per_category_recall", "sanity_check", "inference_seconds"):
        assert key not in run.meta
    assert run.meta["blind"] is True
    assert run.latents is None and run.feature_errors is None
    assert not list(blind_path.glob("*.npy"))


def test_non_blind_run_keeps_the_shared_keys(run_path):
    run = load_run(run_path)
    assert set(run.scores.part) == {"calib", "test"}
    assert run.meta["blind"] is False
    for key in (
        "n_parameters",
        "train_seconds",
        "inference_seconds",
        "calib_alert_rate",
        "window_length",
        "latent_dim",
        "test_metrics",
        "per_category_recall",
        "sanity_check",
    ):
        assert key in run.meta


def test_both_modes_record_the_selection_inputs(run_path, blind_path):
    for path in (run_path, blind_path):
        meta = load_run(path).meta
        assert meta["best_tune_loss"] == pytest.approx(
            pd.read_csv(path / "history.csv").tune_loss.min()
        )
        cells = [[c["magnitude"], c["n_features"]] for c in meta["tune_sanity"]]
        assert cells == [list(g) for g in SANITY_GRID]
        assert meta["threads"] >= 1
        assert (meta["hidden_dim"], meta["dropout"], meta["noise_std"]) == (4, 0.1, 0.1)


def test_blind_run_refuses_results_dir(mod, inputs):
    parts, features, fit, tune = inputs
    with pytest.raises(ValueError, match="blind"):
        mod.run_seed(
            parts,
            features,
            fit,
            tune,
            0,
            2,
            1e-3,
            out_root=mod.RESULTS_DIR / "x",
            blind=True,
            **SETTINGS,
        )
