"""Windowed MLP autoencoder: a feedforward control for the value of temporal input.

It sees the same windows as the LSTM and CNN but flattens them, so it has no
recurrence or convolution. Any gain of those models over this one is therefore
due to how they use temporal structure, not to having more minutes as input.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class WindowedMLPAutoencoder(nn.Module):
    """Reconstructs a whole window of ``window_length`` minutes from a flat vector.

    Input and output are float32 tensors shaped ``(batch, window_length,
    n_features)``, as ``train_autoencoder`` requires. The window is flattened to
    ``window_length * n_features`` values, squeezed through a ``latent_dim``
    bottleneck, and reshaped back.
    """

    def __init__(
        self,
        window_length: int,
        n_features: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        if min(window_length, n_features, hidden_dim, latent_dim) < 1:
            raise ValueError(
                "window_length, n_features, hidden_dim and latent_dim must be positive"
            )
        self.window_length = window_length
        self.n_features = n_features
        flat_dim = window_length * n_features
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, W, F)`` -> latent ``(batch, latent_dim)``."""
        return self.encoder(x.reshape(x.shape[0], -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, W, F)`` -> reconstruction ``(batch, W, F)``."""
        return self.decoder(self.encode(x)).reshape(x.shape)


def embed_windows(
    model: WindowedMLPAutoencoder, x: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    """Latent embedding of each window, row-aligned with ``score_autoencoder``'s output.

    This is what ``save_run(..., latents=...)`` expects, for the latent-space chart.
    """
    model.eval()
    inputs = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        latents = [model.encode(batch).numpy() for batch in inputs.split(batch_size)]
    return np.concatenate(latents, axis=0)
