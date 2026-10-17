"""One training loop and one scoring rule for every neural autoencoder.

Sharing them is what keeps the MLP, LSTM and CNN comparison fair: same loss,
same early stopping, same seeds, same "final minute" score. A model owner only
writes the ``nn.Module``.

The ``nn.Module`` contract: its input is a float32 tensor shaped ``(batch,
features)`` or ``(batch, length, features)``, and it must return a tensor of the
SAME shape. A CNN therefore transposes to channels-first internally and back
before returning. Training loss is the MSE over the whole input or window;
scoring uses the final minute only (as in the proposal).
"""

from __future__ import annotations

import random
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """Seed Python's ``random``, NumPy and PyTorch, so a recorded seed reproduces a run.

    Call it before building the model, so weight initialisation is seeded too.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _input_tensor(x: np.ndarray) -> torch.Tensor:
    """Copy inputs so training and scoring cannot mutate the caller's arrays."""
    if x.ndim not in (2, 3) or any(size == 0 for size in x.shape[1:]):
        raise ValueError("Expected (rows, features) or (windows, length, features)")
    return torch.tensor(x, dtype=torch.float32)


def _reconstruct(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    reconstruction = model(batch)
    if reconstruction.shape != batch.shape:
        raise ValueError("The model must return the same shape as its input")
    return reconstruction


@dataclass(frozen=True)
class TrainResult:
    """What a training run returns.

    Attributes:
        model: the model with its best-epoch weights restored.
        history: one row per epoch run; columns ``epoch`` (from 1),
            ``train_loss``, ``tune_loss`` (mean MSE). This is the loss curve the
            report shows. ``train_loss`` is the mean training loss over the
            epoch's mini-batches, weighted by batch size.
        best_epoch: the epoch with the lowest ``tune_loss``.
        seconds: wall-clock training time, for the runtime comparison, measured
            with ``time.perf_counter()``.
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
    best-epoch weights. The caller's ``train`` and ``tune`` arrays are never
    modified or reordered; shuffling happens on indices.
    """
    if epochs < 1 or batch_size < 1 or patience < 1 or lr < 0:
        raise ValueError("epochs, batch_size and patience must be positive; lr nonnegative")
    train_tensor, tune_tensor = _input_tensor(train), _input_tensor(tune)
    if len(train_tensor) == 0 or len(tune_tensor) == 0:
        raise ValueError("Training and tuning inputs must be nonempty")
    if train_tensor.shape[1:] != tune_tensor.shape[1:]:
        raise ValueError("Training and tuning inputs must have matching sample shapes")

    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        # Permute whole samples/windows, preserving the time axis inside each window.
        order = torch.randperm(len(train_tensor), generator=generator)
        for indices in order.split(batch_size):
            batch = train_tensor[indices].to(device)
            optimizer.zero_grad()
            loss = criterion(_reconstruct(model, batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite")
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(batch)

        model.eval()
        total_tune_loss = 0.0
        with torch.no_grad():
            for batch in tune_tensor.split(batch_size):
                batch = batch.to(device)
                loss = criterion(_reconstruct(model, batch), batch)
                total_tune_loss += loss.item() * len(batch)

        train_loss = total_train_loss / len(train_tensor)
        tune_loss = total_tune_loss / len(tune_tensor)
        if not np.isfinite(tune_loss):
            raise ValueError("Tuning loss is not finite")
        history.append({"epoch": epoch, "train_loss": train_loss, "tune_loss": tune_loss})
        if tune_loss < best_loss:
            best_loss = tune_loss
            best_epoch = epoch
            # state_dict alone shares storage with weights that keep changing.
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    # Device work must finish before measuring elapsed wall-clock time.
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    elif target.type == "mps":
        torch.mps.synchronize()
    return TrainResult(model, pd.DataFrame(history), best_epoch, time.perf_counter() - started)


def score_autoencoder(
    model: nn.Module, x: np.ndarray, *, batch_size: int = 1024, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, feature_errors)`` for each row or window of ``x``.

    ``feature_errors`` is the squared reconstruction error of the final minute,
    shape ``(n, features)``: ``x`` itself for 2-D input, ``x[:, -1, :]`` for
    windows. ``scores`` is its mean over features. Only the final minute counts,
    so a window model's alert refers to minute t, as a single-minute model's does.
    Puts ``model`` in eval mode (dropout off) before scoring.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    inputs = _input_tensor(x)
    model.to(device)
    model.eval()
    if len(inputs) == 0:
        return np.empty(0, dtype=np.float32), np.empty((0, x.shape[-1]), dtype=np.float32)

    errors = []
    with torch.no_grad():
        for batch in inputs.split(batch_size):
            batch = batch.to(device)
            reconstruction = _reconstruct(model, batch)
            if batch.ndim == 3:
                batch = batch[:, -1, :]
                reconstruction = reconstruction[:, -1, :]
            errors.append((reconstruction - batch).square().cpu().numpy())
    feature_errors = np.concatenate(errors, axis=0)
    return feature_errors.mean(axis=1), feature_errors
