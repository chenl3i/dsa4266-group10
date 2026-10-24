"""A convolutional autoencoder for short traffic windows."""

from __future__ import annotations

import torch
from torch import nn


class CNNAutoencoder(nn.Module):
    """Reconstruct windows while compressing their temporal resolution and channels."""

    def __init__(
        self,
        n_features: int,
        window_length: int = 20,
        hidden_channels: int = 32,
        latent_channels: int = 16,
    ) -> None:
        super().__init__()
        if min(n_features, window_length, hidden_channels, latent_channels) < 1:
            raise ValueError("Feature, window, and channel sizes must be positive")

        self.n_features = n_features
        self.window_length = window_length
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.upsample = nn.ConvTranspose1d(
            hidden_channels, n_features, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected windows shaped (batch, length, features)")
        if x.shape[1] != self.window_length or x.shape[2] != self.n_features:
            raise ValueError(
                f"Expected windows shaped (batch, {self.window_length}, {self.n_features})"
            )

        encoded = self.encoder(x.transpose(1, 2))
        decoded = self.decoder(encoded)
        reconstructed = self.upsample(
            decoded,
            output_size=(x.shape[0], self.n_features, self.window_length),
        )
        return reconstructed.transpose(1, 2)
