"""Sliding windows of consecutive minutes for the LSTM, CNN and flattened-MLP models.

A window ending at minute t holds t and the minutes just before it, never later
ones, and never spans a gap: a window that jumps the April outage would show the
model a "sequence" that never happened. Every model, including the single-minute
ones, is scored on the same ``eligible_positions`` so the comparison uses the
same minutes.
"""

from __future__ import annotations

import numpy as np


def eligible_positions(segment: np.ndarray, length: int) -> np.ndarray:
    """Row positions t where rows ``t-length+1 .. t`` all share one segment id.

    ``segment`` is ``Part.segment.to_numpy()``. Returns int64 positions in
    increasing order. With ``length == 1`` every row is eligible.
    """
    segment = np.asarray(segment)
    if segment.ndim != 1:
        raise ValueError("segment must be one-dimensional")
    if length < 1:
        raise ValueError("length must be at least 1")
    if segment.size == 0:
        return np.empty(0, dtype=np.int64)

    boundaries = np.flatnonzero(segment[1:] != segment[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [segment.size]))
    eligible = [
        np.arange(start + length - 1, end, dtype=np.int64)
        for start, end in zip(starts, ends, strict=True)
        if end - start >= length
    ]
    if not eligible:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(eligible)


def make_windows(X: np.ndarray, segment: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack the windows that end at each eligible position.

    Returns ``(windows, positions)``: ``windows[i]`` is ``X[positions[i]-length+1 :
    positions[i]+1]``, shape ``(m, length, features)``. When nothing is eligible,
    returns shapes ``(0, length, features)`` and ``(0,)``.
    """
    X = np.asarray(X)
    segment = np.asarray(segment)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if segment.ndim != 1 or len(segment) != len(X):
        raise ValueError("segment must be one-dimensional and match the rows in X")

    positions = eligible_positions(segment, length)
    if positions.size == 0:
        return np.empty((0, length, X.shape[1]), dtype=X.dtype), positions

    windows = np.stack([X[position - length + 1 : position + 1] for position in positions])
    return windows, positions
