"""Recurrent window autoencoder (LSTM or GRU).

It tests whether the order of minutes inside a window carries information that the
single-minute MLP cannot see, which is the question the proposal asks of a sequence
model. LSTM and GRU share one class so the only difference between them is the cell.
"""

from __future__ import annotations

import torch
from torch import nn

_CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}


class LSTMAutoencoder(nn.Module):
    """Encode a window into one latent vector, then decode it back to the window.

    The decoder repeats the latent vector once per minute of the input, so the
    model has no ``window_length`` argument and any window length works. The
    bottleneck is a single vector, so the model cannot copy the window through.

    Args:
        n_features: features per minute.
        hidden_dim: size of the recurrent state in encoder and decoder.
        latent_dim: size of the bottleneck.
        cell: ``"lstm"`` or ``"gru"``. The run script names its results folder
            after it, so the two are never mixed up.
    """

    def __init__(
        self, n_features: int, hidden_dim: int = 64, latent_dim: int = 16, cell: str = "lstm"
    ) -> None:
        super().__init__()
        if n_features < 1 or hidden_dim < 1 or latent_dim < 1:
            raise ValueError("n_features, hidden_dim and latent_dim must be positive")
        if cell not in _CELLS:
            raise ValueError(f"cell must be one of {sorted(_CELLS)}, got {cell!r}")
        self.n_features = n_features
        self.cell = cell
        rnn_cls = _CELLS[cell]
        self.encoder = rnn_cls(n_features, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = rnn_cls(latent_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, length, features)`` to ``(batch, latent_dim)``."""
        hidden = self.encoder(x)[1]
        if self.cell == "lstm":
            hidden = hidden[0]  # (h_n, c_n): the bottleneck reads h_n only
        return self.to_latent(hidden[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        repeated = latent.unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.output(self.decoder(repeated)[0])
