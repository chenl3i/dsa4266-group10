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

    Two regularisers fight the memorising that a first layer this wide invites.
    ``dropout`` is applied to the encoder's hidden layer. ``noise_std`` adds
    Gaussian noise to the input, in units of fit-part standard deviations. Both
    act only in training mode. ``train_autoencoder`` scores the output against
    the clean batch, so noise makes this a denoising autoencoder without any
    change to the shared training loop. Tuning loss and scoring run in eval mode,
    so they always see clean inputs and are comparable across settings.
    """

    def __init__(
        self,
        window_length: int,
        n_features: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        dropout: float = 0.0,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        if min(window_length, n_features, hidden_dim, latent_dim) < 1:
            raise ValueError(
                "window_length, n_features, hidden_dim and latent_dim must be positive"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if noise_std < 0.0:
            raise ValueError("noise_std must be nonnegative")
        self.window_length = window_length
        self.n_features = n_features
        self.noise_std = noise_std
        flat_dim = window_length * n_features
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        noisy = x + self.noise_std * torch.randn_like(x) if self.training else x
        return self.decoder(self.encode(noisy)).reshape(x.shape)


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
