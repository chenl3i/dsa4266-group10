"""Single-minute nonlinear reconstruction baseline, trained by the shared loop."""

from __future__ import annotations

import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    """Encode each minute independently and reconstruct its scaled features.

    Input and output have shape ``(batch, n_features)``. Use the feature count
    after preprocessing as n_features. Hidden widths and latent_dim are starting
    settings to tune, not experimentally selected values. The decoder mirrors
    the encoder and ends with a linear layer because scaled counts can be negative.
    """

    def __init__(
        self, n_features: int, hidden_dims: tuple[int, ...] = (64, 32), latent_dim: int = 8
    ) -> None:
        super().__init__()
        widths = (n_features, *hidden_dims, latent_dim)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in widths):
            raise ValueError("All layer dimensions must be positive integers")
        if latent_dim >= n_features:
            raise ValueError("latent_dim must be smaller than n_features")
        self.n_features = n_features

        def layers(dimensions: tuple[int, ...]) -> nn.Sequential:
            modules = []
            for i, (source, target) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
                modules.append(nn.Linear(source, target))
                if i < len(dimensions) - 2:
                    modules.append(nn.ReLU())
            return nn.Sequential(*modules)

        self.encoder = layers(widths)
        self.decoder = layers(tuple(reversed(widths)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError("Expected single-minute inputs shaped (batch, n_features)")
        return self.decoder(self.encoder(x))
