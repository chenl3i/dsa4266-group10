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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_lstm.py"
_spec = importlib.util.spec_from_file_location("run_lstm", SCRIPT)
run_lstm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_lstm)

WINDOW = 5
SETTINGS = dict(window_length=WINDOW, epochs=2, hidden_dim=8, latent_dim=2)


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
    return run_lstm.run(parts, seed=0, out_root=tmp_path_factory.mktemp("out"), **SETTINGS)


def test_writes_a_loadable_run(run_path):
    run = load_run(run_path)
    assert run.meta["model"] == "lstm"
    assert run.meta["seed"] == 0
    assert run.meta["window_length"] == WINDOW
    assert run_path.parts[-2:] == ("lstm", "seed0")
    assert (run_path / "history.csv").exists()


def test_threshold_is_the_calib_quantile(run_path):
    run = load_run(run_path)
    calib = run.scores[run.scores.part == "calib"].score
    # scores.csv keeps 9 significant digits, so the reloaded quantile matches to ~1e-8.
    assert run.meta["threshold"] == pytest.approx(fit_threshold(calib), rel=1e-6)


def test_scores_cover_only_eligible_final_minutes(run_path, parts):
    scores = load_run(run_path).scores
    for name, part in parts.items():
        positions = eligible_positions(part.segment.to_numpy(), WINDOW)
        rows = scores[scores.part == name]
        assert len(rows) == len(positions)
        assert (rows.timestamp.to_numpy() == part.X.index[positions].to_numpy()).all()
    assert len(scores[scores.part == "fit"]) == 52


def test_arrays_align_with_scores(run_path):
    run = load_run(run_path)
    assert run.latents.shape == (len(run.scores), 2)
    assert run.feature_errors.shape == (len(run.scores), 3)
    order = run.scores.part.drop_duplicates().tolist()
    assert order == ["fit", "tune", "calib", "test"]


def test_gru_writes_under_gru(parts, tmp_path):
    path = run_lstm.run(parts, seed=0, out_root=tmp_path, cell="gru", **SETTINGS)
    assert path == tmp_path / "gru" / "seed0"


def test_too_long_window_raises(parts, tmp_path):
    with pytest.raises(ValueError, match="fit"):
        run_lstm.run(parts, seed=0, out_root=tmp_path, **{**SETTINGS, "window_length": 40})


def test_meta_has_the_comparison_table_keys(run_path):
    meta = load_run(run_path).meta
    for key in ("threshold", "n_parameters", "train_seconds", "inference_seconds"):
        assert meta[key] is not None


def test_meta_records_settings_and_history(run_path):
    run = load_run(run_path)
    meta = run.meta
    assert (meta["n_features"], meta["epochs"], meta["patience"], meta["device"]) == (
        3,
        2,
        5,
        "cpu",
    )
    assert len(pd.read_csv(run_path / "history.csv")) == meta["epochs_run"]
    calib = run.scores[run.scores.part == "calib"].score
    assert meta["calib_alert_rate"] == pytest.approx((calib > meta["threshold"]).mean())


def test_test_labels_align_with_scored_minutes(run_path, parts):
    meta = load_run(run_path).meta
    test = parts["test"]
    positions = eligible_positions(test.segment.to_numpy(), WINDOW)
    expected = int(test.y.to_numpy()[positions].sum())
    assert expected > 0
    assert meta["test_metrics"]["tp"] + meta["test_metrics"]["fn"] == expected
    by_category = {r["category"]: r for r in meta["per_category_recall"]}
    assert by_category[ATTACK_LABELS[0]]["n_positive"] == expected


def test_meta_records_sanity_check(run_path):
    checks = load_run(run_path).meta["sanity_check"]
    # JSON turns the grid's tuples into lists. The 30-row calib picks no rows at
    # rate 0.01, so auc is nan here; only the structure is checked.
    assert [[c["magnitude"], c["n_features"]] for c in checks] == [list(g) for g in SANITY_GRID]
    assert all("auc" in c for c in checks)


@pytest.fixture(scope="module")
def blind_path(parts, tmp_path_factory):
    out = tmp_path_factory.mktemp("blind")
    return run_lstm.run(parts, seed=0, out_root=out, blind=True, **SETTINGS)


def test_blind_run_never_touches_test(blind_path):
    run = load_run(blind_path)
    assert "test" not in set(run.scores.part)
    for key in ("test_metrics", "per_category_recall", "sanity_check", "inference_seconds"):
        assert key not in run.meta
    assert run.meta["blind"] is True
    assert run.latents is None and run.feature_errors is None


def test_both_modes_record_the_selection_inputs(run_path, blind_path):
    for path in (run_path, blind_path):
        meta = load_run(path).meta
        assert meta["best_tune_loss"] == pytest.approx(
            pd.read_csv(path / "history.csv").tune_loss.min()
        )
        cells = [[c["magnitude"], c["n_features"]] for c in meta["tune_sanity"]]
        assert cells == [list(g) for g in SANITY_GRID]
        assert {"auc", "ap", "recall"} <= set(meta["tune_sanity"][0])
        assert meta["threads"] >= 1


def test_blind_run_refuses_results_dir(parts):
    with pytest.raises(ValueError, match="blind"):
        run_lstm.run(parts, seed=0, out_root=run_lstm.RESULTS_DIR / "x", blind=True, **SETTINGS)
