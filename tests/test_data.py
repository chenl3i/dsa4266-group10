import pandas as pd
import pytest

from ugr16ad import data


def assert_shared_index(part):
    assert part.X.index.name == "timestamp"
    assert isinstance(part.X.index, pd.DatetimeIndex)
    for field in (part.y, part.labels, part.segment):
        assert field.index.equals(part.X.index)
        assert field.index.name == "timestamp"
    assert pd.api.types.is_integer_dtype(part.segment)


def test_attack_labels_exclude_blacklist():
    assert len(data.ATTACK_LABELS) == 7
    assert data.BLACKLIST_LABEL not in data.ATTACK_LABELS


def test_read_split_indexes_by_timestamp_and_drops_key(raw_dir):
    X, Y = data.read_split("train", directory=raw_dir)
    assert data.KEY not in X.columns
    assert data.KEY not in Y.columns
    assert X.index.name == "timestamp"
    assert X.index[0] == pd.Timestamp("2016-03-19 00:00")
    assert X.index.equals(Y.index)
    assert list(X.columns) == ["f_a", "f_b", "f_const"]


def test_read_split_rejects_misaligned_keys(raw_dir):
    path = raw_dir / "UGR16v2.Ytrain.csv"
    Y = pd.read_csv(path)
    # Order-preserving nudge (00:13 -> 00:14): only an X-vs-Y equality check catches it.
    Y.loc[9, "Row"] += 1
    Y.to_csv(path, index=False)
    with pytest.raises(ValueError):
        data.read_split("train", directory=raw_dir)


def test_read_split_rejects_duplicate_keys(raw_dir):
    for prefix in ("X", "Y"):
        path = raw_dir / f"UGR16v2.{prefix}train.csv"
        frame = pd.read_csv(path)
        pd.concat([frame, frame.tail(1)]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        data.read_split("train", directory=raw_dir)


def test_read_split_rejects_unsorted_keys(raw_dir):
    # Swapped identically in X and Y: aligned and duplicate-free, only unsorted.
    for prefix in ("X", "Y"):
        path = raw_dir / f"UGR16v2.{prefix}train.csv"
        frame = pd.read_csv(path)
        order = [1, 0, *range(2, len(frame))]
        frame.iloc[order].to_csv(path, index=False)
    with pytest.raises(ValueError):
        data.read_split("train", directory=raw_dir)


def test_segment_ids_increment_at_gaps():
    index = pd.DatetimeIndex(
        [
            "2016-03-19 00:00",
            "2016-03-19 00:01",
            "2016-03-19 00:02",
            "2016-03-19 00:10",
            "2016-03-19 00:11",
            "2016-03-19 00:13",
        ]
    )
    # Two gaps, the second only 2 minutes: a cummax would give every row after the first gap id 1.
    ids = data.segment_ids(index)
    assert ids.tolist() == [0, 0, 0, 1, 1, 2]
    assert ids.index.equals(index)


def test_make_part_target_ignores_blacklist(raw_dir):
    part = data.make_part(*data.read_split("test", directory=raw_dir))
    assert part.y.dtype == bool
    assert part.y.tolist() == [False, False, True, False, True]
    assert data.BLACKLIST_LABEL not in part.labels.columns
    assert list(part.labels.columns) == list(data.ATTACK_LABELS)
    assert part.segment.tolist() == [0, 0, 0, 0, 0]
    assert_shared_index(part)


def test_load_dataset_splits_chronologically(raw_dir):
    parts = data.load_dataset(directory=raw_dir, fit_rows=6, tune_fraction=0.5)
    assert list(parts) == ["fit", "tune", "calib", "test"]
    assert [len(p.X) for p in parts.values()] == [6, 2, 2, 5]
    assert parts["fit"].X.index.max() < parts["tune"].X.index.min()
    assert parts["tune"].X.index.max() < parts["calib"].X.index.min()


def test_load_dataset_floors_tune_count(raw_dir):
    # The 4 rows after fit: int(4 * 0.4) = int(1.6) = 1 tune row, not round() = 2.
    parts = data.load_dataset(directory=raw_dir, fit_rows=6, tune_fraction=0.4)
    assert [len(p.X) for p in parts.values()] == [6, 1, 3, 5]
    assert parts["fit"].X.index.max() < parts["tune"].X.index.min()
    assert parts["tune"].X.index.max() < parts["calib"].X.index.min()


def test_segment_restarts_per_part(raw_dir):
    # Parts are windowed separately, so a window in tune can never reach into fit.
    parts = data.load_dataset(directory=raw_dir, fit_rows=6, tune_fraction=0.5)
    for part in parts.values():
        assert part.segment.iloc[0] == 0
        assert_shared_index(part)


@pytest.mark.slow
def test_real_data_matches_proposal():
    parts = data.load_dataset()
    assert len(parts["fit"].X) == 78_609
    assert len(parts["tune"].X) + len(parts["calib"].X) == 19_653
    assert len(parts["test"].X) == 43_200
    assert parts["fit"].X.shape[1] == 134
    assert not any(parts[name].y.any() for name in ("fit", "tune", "calib"))
    assert int(parts["test"].y.sum()) == 1_364
