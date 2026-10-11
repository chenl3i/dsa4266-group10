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
    raise NotImplementedError("Kesha, week 1: see docs/team-plan.md")


def make_windows(X: np.ndarray, segment: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack the windows that end at each eligible position.

    Returns ``(windows, positions)``: ``windows[i]`` is ``X[positions[i]-length+1 :
    positions[i]+1]``, shape ``(m, length, features)``. When nothing is eligible,
    returns shapes ``(0, length, features)`` and ``(0,)``.
    """
    raise NotImplementedError("Kesha, week 1: see docs/team-plan.md")
