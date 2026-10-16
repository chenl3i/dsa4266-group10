import json

import numpy as np
import pandas as pd
import pytest

from ugr16ad.results import REQUIRED_META, SCORE_COLUMNS, load_run, run_dir, save_run

SCORES = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(["2016-07-28 00:00", "2016-07-28 00:01", "2016-07-28 00:02"]),
        "part": ["test", "test", "test"],
        "score": [0.1, 2.5, 0.3],
    }
)
META = {"model": "pca", "seed": 0, "threshold": 1.5}


def test_run_dir_layout(tmp_path):
    assert run_dir("pca", 0, root=tmp_path) == tmp_path / "pca" / "seed0"


def test_run_dir_does_not_create_the_directory(tmp_path):
    run_dir("pca", 0, root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_save_then_load_round_trips(tmp_path):
    latents = np.arange(6, dtype=np.float32).reshape(3, 2)
    errors = np.ones((3, 4), dtype=np.float32)
    save_run(tmp_path, scores=SCORES, meta=META, latents=latents, feature_errors=errors)
    run = load_run(tmp_path)
    pd.testing.assert_frame_equal(run.scores, SCORES, check_dtype=False)
    assert pd.api.types.is_datetime64_any_dtype(run.scores["timestamp"])
    assert run.meta == META
    np.testing.assert_array_equal(run.latents, latents)
    np.testing.assert_array_equal(run.feature_errors, errors)


def test_meta_values_keep_their_types_and_extra_keys_survive(tmp_path):
    # 0 == 0.0 in Python, so equality alone would not notice a seed that came back as a float.
    meta = {**META, "settings": {"n_components": 8}, "train_seconds": 1.25}
    save_run(tmp_path, scores=SCORES, meta=meta)
    loaded = load_run(tmp_path).meta
    assert loaded == meta
    assert type(loaded["seed"]) is int
    assert type(loaded["threshold"]) is float
    assert json.loads((tmp_path / "meta.json").read_text()) == meta


def test_numpy_scalars_in_meta_are_written_as_plain_python_types(tmp_path):
    meta = {**META, "threshold": np.float32(1.5), "seed": np.int64(0)}
    save_run(tmp_path, scores=SCORES, meta=meta)
    loaded = load_run(tmp_path).meta
    assert loaded["threshold"] == 1.5
    assert type(loaded["threshold"]) is float
    assert type(loaded["seed"]) is int


def test_save_run_creates_missing_parent_directories(tmp_path):
    directory = tmp_path / "reports" / "results" / "pca" / "seed0"
    save_run(directory, scores=SCORES, meta=META)
    assert (directory / "scores.csv").is_file()
    assert (directory / "meta.json").is_file()


def test_optional_arrays_load_as_none(tmp_path):
    save_run(tmp_path, scores=SCORES, meta=META)
    run = load_run(tmp_path)
    assert run.latents is None
    assert run.feature_errors is None
    assert not (tmp_path / "latents.npy").exists()
    assert not (tmp_path / "feature_errors.npy").exists()


def test_each_optional_array_is_stored_and_loaded_on_its_own(tmp_path):
    latents = np.arange(6, dtype=np.float32).reshape(3, 2)
    errors = np.ones((3, 4), dtype=np.float32)
    only_latents = tmp_path / "a"
    save_run(only_latents, scores=SCORES, meta=META, latents=latents)
    run = load_run(only_latents)
    np.testing.assert_array_equal(run.latents, latents)
    assert run.feature_errors is None
    only_errors = tmp_path / "b"
    save_run(only_errors, scores=SCORES, meta=META, feature_errors=errors)
    run = load_run(only_errors)
    assert run.latents is None
    np.testing.assert_array_equal(run.feature_errors, errors)


@pytest.mark.parametrize("column", SCORE_COLUMNS)
def test_save_run_rejects_missing_score_column(tmp_path, column):
    with pytest.raises(ValueError):
        save_run(tmp_path, scores=SCORES.drop(columns=column), meta=META)


@pytest.mark.parametrize("key", REQUIRED_META)
def test_save_run_rejects_meta_missing_a_required_key(tmp_path, key):
    meta = {k: v for k, v in META.items() if k != key}
    with pytest.raises(ValueError):
        save_run(tmp_path, scores=SCORES, meta=meta)


def test_save_run_rejects_misaligned_latents(tmp_path):
    with pytest.raises(ValueError):
        save_run(tmp_path, scores=SCORES, meta=META, latents=np.zeros((2, 2)))


def test_save_run_rejects_misaligned_feature_errors(tmp_path):
    with pytest.raises(ValueError):
        save_run(tmp_path, scores=SCORES, meta=META, feature_errors=np.zeros((4, 3)))
