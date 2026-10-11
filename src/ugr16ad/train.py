"""One training loop and one scoring rule for every neural autoencoder.

Sharing them is what keeps the MLP, LSTM and CNN comparison fair: same loss,
same early stopping, same seeds, same "final minute" score. A model owner only
writes the ``nn.Module``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch import nn


def set_seed(seed: int) -> None:
    """Seed Python's ``random``, NumPy and PyTorch, so a recorded seed reproduces a run.

    Call it before building the model, so weight initialisation is seeded too.
    """
    raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")


@dataclass(frozen=True)
class TrainResult:
    """What a training run returns.

    Attributes:
        model: the model with its best-epoch weights restored.
        history: one row per epoch run; columns ``epoch`` (from 1),
            ``train_loss``, ``tune_loss`` (mean MSE). This is the loss curve the
            report shows.
        best_epoch: the epoch with the lowest ``tune_loss``.
        seconds: wall-clock training time, for the runtime comparison.
    """

    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    seconds: float


def train_autoencoder(
    model: nn.Module,
    train: np.ndarray,
    tune: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 0,
    device: str = "cpu",
) -> TrainResult:
    """Fit ``model`` to reconstruct its input (MSE loss, Adam optimiser).

    ``train`` and ``tune`` are 2-D ``(rows, features)`` or 3-D ``(windows,
    length, features)``. Mini-batches are shuffled within the fit part only,
    using ``seed``; that never mixes rows across parts. Stops when ``tune_loss``
    has not strictly improved for ``patience`` epochs, then restores the
    best-epoch weights.
    """
    raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")


def score_autoencoder(
    model: nn.Module, x: np.ndarray, *, batch_size: int = 1024, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, feature_errors)`` for each row or window of ``x``.

    ``feature_errors`` is the squared reconstruction error of the final minute,
    shape ``(n, features)``: ``x`` itself for 2-D input, ``x[:, -1, :]`` for
    windows. ``scores`` is its mean over features. Only the final minute counts,
    so a window model's alert refers to minute t, as a single-minute model's does.
    """
    raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")
