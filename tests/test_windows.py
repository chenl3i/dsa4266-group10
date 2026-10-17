import numpy as np

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
