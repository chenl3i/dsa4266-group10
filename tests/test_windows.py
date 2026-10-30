import numpy as np
import pytest

from ugr16ad.preprocess import Preprocessor
from ugr16ad.windows import eligible_positions, make_windows

SEGMENT = np.array([0, 0, 0, 0, 1, 1, 1])
X = np.arange(7 * 2, dtype=np.float32).reshape(7, 2)


def test_eligible_positions_need_a_full_window_in_one_segment():
    assert eligible_positions(SEGMENT, 3).tolist() == [2, 3, 6]


def test_eligible_positions_are_int64():
    assert eligible_positions(SEGMENT, 3).dtype == np.int64
    assert eligible_positions(SEGMENT, 9).dtype == np.int64


def test_length_one_makes_every_row_eligible():
    assert eligible_positions(SEGMENT, 1).tolist() == list(range(7))


def test_short_segment_between_valid_ones_is_not_jumped():
    # The middle segment (length 2) is shorter than the window; no window may bridge it.
    segment = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    assert eligible_positions(segment, 3).tolist() == [2, 7, 8]


def test_window_needs_every_row_in_the_segment_not_just_both_ends():
    # Rows 0 and 2 share an id but row 1 does not, so position 2 has no valid window.
    assert eligible_positions(np.array([0, 1, 0, 0]), 3).tolist() == []


def test_windows_never_cross_segments():
    windows, positions = make_windows(X, SEGMENT, 3)
    assert windows.shape == (3, 3, 2)
    assert positions.tolist() == [2, 3, 6]
    np.testing.assert_array_equal(windows[0], X[0:3])
    np.testing.assert_array_equal(windows[1], X[1:4])
    np.testing.assert_array_equal(windows[2], X[4:7])
    np.testing.assert_array_equal(windows[:, -1, :], X[positions])


def test_make_windows_keeps_dtype_and_int64_positions():
    windows, positions = make_windows(X, SEGMENT, 3)
    assert windows.dtype == np.float32
    assert positions.dtype == np.int64


def test_window_longer_than_any_segment_gives_empty_result():
    windows, positions = make_windows(X, SEGMENT, 5)
    assert windows.shape == (0, 5, 2)
    assert positions.shape == (0,)
    assert positions.dtype == np.int64


@pytest.mark.slow
def test_real_windows_respect_gaps_and_part_edges(real_parts):
    length = 20
    pre = Preprocessor().fit(real_parts["fit"].X)
    for part in real_parts.values():
        transformed = pre.transform(part.X)
        segment = part.segment.to_numpy()
        windows, positions = make_windows(transformed, segment, length)

        assert (segment[positions - length + 1] == segment[positions]).all()
        np.testing.assert_array_equal(windows[:, -1], transformed[positions])

        # Each part is windowed on its own, so no window reaches back across the DST
        # hour, the 1 to 6 Apr outage or a part edge: the first eligible position of
        # every long-enough segment is its start + 19.
        starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
        sizes = np.diff(np.r_[starts, len(segment)])
        long_enough = sizes >= length
        assert set((starts[long_enough] + length - 1).tolist()) <= set(positions.tolist())
        assert not set((starts[long_enough] + length - 2).tolist()) & set(positions.tolist())
        assert len(positions) == int(np.maximum(0, sizes - length + 1).sum())
