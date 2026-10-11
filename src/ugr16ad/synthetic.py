"""Fake anomalies for the validation sanity check.

Validation has no labelled attacks, so this is the only way to check before
test that a model reacts to anything at all. Inputs are already standardised
with fit-part statistics, so a shift of ``magnitude`` means "this many fit-period
standard deviations" and uses no information from later data.
"""

from __future__ import annotations

import numpy as np


def inject_anomalies(
    X: np.ndarray, *, rate: float, magnitude: float, n_features: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(copy of X with anomalies, bool mask of altered rows)``.

    Picks ``round(rate * len(X))`` rows at random. In each, adds ``magnitude``
    to ``n_features`` randomly chosen columns (different columns per row).
    ``X`` is 2-D ``(rows, features)`` and is never modified in place. The same
    ``seed`` gives the same result.
    """
    raise NotImplementedError("Phone, week 1: see docs/team-plan.md")
