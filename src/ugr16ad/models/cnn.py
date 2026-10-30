"""1D convolutional window autoencoder.

It asks the same question as the LSTM, whether the order of minutes inside a window
helps, but with convolutions instead of recurrence. To make that comparison fair the
CNN shares the LSTM's shape: the whole window is squeezed into one latent vector, and
the final minute (the only one scored) depends on every minute of the window.
"""

from __future__ import annotations

import torch
from torch import nn

# Kernel 3 with these dilations reaches back 1 + 2 * (1 + 2 + 4 + 8) = 31 minutes, so
# the last encoder position covers a 20-minute window. Plain stacked or strided convs
# covered only the last 7 minutes, which made the CNN a shorter-context model.
_DILATIONS = (1, 2, 4, 8)
_KERNEL = 3
_REACH = 1 + (_KERNEL - 1) * sum(_DILATIONS)


def _causal_conv(in_channels: int, out_channels: int, dilation: int) -> nn.Sequential:
    """Pad on the left only, so each position sees itself and earlier minutes."""
    return nn.Sequential(
        nn.ConstantPad1d(((_KERNEL - 1) * dilation, 0), 0.0),
        nn.Conv1d(in_channels, out_channels, _KERNEL, dilation=dilation),
        nn.GELU(),
    )


class CNNAutoencoder(nn.Module):
    """Encode a window into one latent vector, then decode it back to the window.

    The encoder reads the last position of a causal dilated conv stack, which has seen
    the whole window. The decoder expands the latent vector with a linear layer, so
    every output minute, the scored final one included, has its own weights. A
    transposed-conv decoder gave the final minute a single edge tap. The linear layer
    is why the model needs ``window_length`` and rejects other lengths.

    The defaults match the LSTM's bottleneck (``latent_dim=16``) but not its width:
    32 channels give about half the LSTM's parameters (41k vs 82k on 133 features).
    Both were fixed before any test result was seen and are not tuned, so the two
    models get the same (zero) tuning budget.

    Args:
        n_features: features per minute.
        window_length: minutes per window.
        hidden_channels: channels in the conv layers.
        latent_dim: size of the bottleneck, comparable with the LSTM's ``latent_dim``.
    """

    def __init__(
        self,
        n_features: int,
        window_length: int = 20,
        hidden_channels: int = 32,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        if min(n_features, window_length, hidden_channels, latent_dim) < 1:
            raise ValueError("Feature, window, channel and latent sizes must be positive")
        if window_length > _REACH:
            raise ValueError(
                f"window_length {window_length} exceeds the encoder reach of {_REACH} minutes"
            )

        self.n_features = n_features
        self.window_length = window_length
        self.hidden_channels = hidden_channels
        self.encoder = nn.Sequential(
            *(
                _causal_conv(n_features if i == 0 else hidden_channels, hidden_channels, d)
                for i, d in enumerate(_DILATIONS)
            )
        )
        self.to_latent = nn.Linear(hidden_channels, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_channels * window_length)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, _KERNEL, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, n_features, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, length, features)`` to ``(batch, latent_dim)``."""
        if x.ndim != 3:
            raise ValueError("Expected windows shaped (batch, length, features)")
        if x.shape[1] != self.window_length or x.shape[2] != self.n_features:
            raise ValueError(
                f"Expected windows shaped (batch, {self.window_length}, {self.n_features})"
            )
        return self.to_latent(self.encoder(x.transpose(1, 2))[:, :, -1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.from_latent(self.encode(x))
        expanded = expanded.view(-1, self.hidden_channels, self.window_length)
        return self.decoder(expanded).transpose(1, 2)
