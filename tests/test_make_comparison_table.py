import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ugr16ad.data import ATTACK_LABELS
from ugr16ad.results import run_dir, save_run

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_comparison_table.py"
_spec = importlib.util.spec_from_file_location("make_comparison_table", SCRIPT)
table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(table)

DOS, SPAM = ATTACK_LABELS[0], ATTACK_LABELS[-1]
MINUTES = pd.date_range("2016-07-28 00:00", periods=6, freq="min", name="timestamp")


def _labels():
    # Two attack minutes (positions 2 and 4), both DoS; spam never occurs.
    labels = pd.DataFrame(0, index=MINUTES, columns=list(ATTACK_LABELS))
    labels.loc[MINUTES[2], DOS] = 1
    labels.loc[MINUTES[4], DOS] = 2
    return labels


LABELS = _labels()
Y_TRUE = (LABELS > 0).any(axis=1).to_numpy()
# At threshold 0.5, PERFECT alerts on positions 2 and 4 (tp=2, fp=0).
PERFECT = [0.1, 0.2, 0.9, 0.3, 0.8, 0.1]
# MIXED alerts on positions 0 and 2 (tp=1, fp=1, fn=1, tn=3).
MIXED = [0.9, 0.2, 0.9, 0.3, 0.1, 0.1]


def write_run(root, model, seed, scores, minutes=MINUTES, threshold=0.5):
    frame = pd.DataFrame({"timestamp": minutes, "part": "test", "score": scores})
    meta = {
        "model": model,
        "seed": seed,
        "threshold": threshold,
        "train_seconds": 2.0,
        "inference_seconds": 0.5,
        "n_parameters": 10,
    }
    save_run(run_dir(model, seed, root=root), scores=frame, meta=meta)


def test_runs_scored_on_different_numbers_of_minutes_stop_the_table(tmp_path):
    write_run(tmp_path, "a", 0, PERFECT)
    write_run(tmp_path, "b", 0, MIXED[:-1], minutes=MINUTES[:-1])
    with pytest.raises(SystemExit, match="different test minutes"):
        table.common_minutes(table.load_runs(tmp_path))


def test_runs_scored_on_shifted_minutes_stop_the_table(tmp_path):
    write_run(tmp_path, "a", 0, PERFECT)
    write_run(tmp_path, "b", 0, MIXED, minutes=MINUTES + pd.Timedelta(minutes=1))
    with pytest.raises(SystemExit, match="different test minutes"):
        table.common_minutes(table.load_runs(tmp_path))


def test_the_error_names_the_mismatched_run(tmp_path):
    write_run(tmp_path, "a", 0, PERFECT)
    write_run(tmp_path, "b", 0, MIXED[:-1], minutes=MINUTES[:-1])
    with pytest.raises(SystemExit) as error:
        table.common_minutes(table.load_runs(tmp_path))
    assert "('b', 0)" in str(error.value)


def test_no_runs_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="No runs found"):
        table.load_runs(tmp_path)


def test_matching_runs_produce_the_expected_rows(tmp_path):
    write_run(tmp_path, "a", 0, PERFECT)
    write_run(tmp_path, "b", 0, MIXED)
    runs = table.load_runs(tmp_path)
    common = table.common_minutes(runs)
    assert common.equals(MINUTES)

    per_seed, summary, per_category = table.build_tables(runs, LABELS.loc[common], Y_TRUE)

    assert list(summary.index) == ["a", "b"]
    assert summary.loc["a", "recall_mean"] == 1.0
    assert summary.loc["a", "precision_mean"] == 1.0
    assert summary.loc["a", "fpr_mean"] == 0.0
    assert summary.loc["a", "average_precision_mean"] == pytest.approx(1.0)
    assert summary.loc["b", "recall_mean"] == 0.5
    assert summary.loc["b", "precision_mean"] == 0.5
    assert summary.loc["b", "fpr_mean"] == pytest.approx(0.25)
    assert summary.loc["b", "alert_rate_mean"] == pytest.approx(2 / 6)
    assert summary.loc["a", "train_seconds_mean"] == 2.0
    assert summary.loc["a", "n_parameters"] == 10
    assert len(per_seed) == 2

    assert per_category.loc[("a", DOS), "n_positive"] == 2
    assert per_category.loc[("a", DOS), "recall"] == 1.0
    assert per_category.loc[("b", DOS), "recall"] == 0.5
    # No positives means recall is undefined, never a claimed 0%.
    assert per_category.loc[("a", SPAM), "n_positive"] == 0
    assert np.isnan(per_category.loc[("a", SPAM), "recall"])


def test_seeds_of_one_model_are_averaged(tmp_path):
    write_run(tmp_path, "a", 0, PERFECT)
    write_run(tmp_path, "a", 1, MIXED)
    _, summary, _ = table.build_tables(table.load_runs(tmp_path), LABELS, Y_TRUE)

    assert list(summary.index) == ["a"]
    assert summary.loc["a", "recall_mean"] == pytest.approx(0.75)
    assert summary.loc["a", "recall_std"] == pytest.approx(np.std([1.0, 0.5], ddof=1))


def test_each_run_is_judged_at_its_own_frozen_threshold(tmp_path):
    # The same scores alert on every minute at threshold 0.0 and on none at 1.0.
    write_run(tmp_path, "low", 0, PERFECT, threshold=0.0)
    write_run(tmp_path, "high", 0, PERFECT, threshold=1.0)
    _, summary, _ = table.build_tables(table.load_runs(tmp_path), LABELS, Y_TRUE)

    assert summary.loc["low", "alert_rate_mean"] == 1.0
    assert summary.loc["high", "alert_rate_mean"] == 0.0
    assert summary.loc["high", "precision_mean"] == 0.0


def test_main_writes_the_three_tables(tmp_path, monkeypatch):
    write_run(tmp_path, "a", 0, PERFECT)
    write_run(tmp_path, "b", 0, MIXED)
    features = pd.DataFrame({"f": range(6)}, index=MINUTES)
    monkeypatch.setattr(table, "read_split", lambda split: (features, LABELS))

    table.main(tmp_path)

    summary = pd.read_csv(tmp_path / "comparison_summary.csv", index_col=0)
    assert summary.loc["a", "recall_mean"] == 1.0
    assert summary.loc["b", "recall_mean"] == 0.5
    per_category = pd.read_csv(tmp_path / "comparison_per_category.csv")
    assert set(per_category["category"]) == set(ATTACK_LABELS)
    assert (tmp_path / "comparison_per_seed.csv").exists()


def test_main_stops_before_writing_when_runs_disagree(tmp_path):
    write_run(tmp_path, "a", 0, PERFECT)
    write_run(tmp_path, "b", 0, MIXED[:-1], minutes=MINUTES[:-1])
    with pytest.raises(SystemExit, match="different test minutes"):
        table.main(tmp_path)
    assert not (tmp_path / "comparison_summary.csv").exists()
