import numpy as np
import pytest
import torch

from ugr16ad.models.cnn_autoencoder import CNNAutoencoder
from ugr16ad.train import score_autoencoder


def test_cnn_reconstructs_twenty_minute_windows_with_matching_shape():
    model = CNNAutoencoder(n_features=5)
    windows = torch.randn(4, 20, 5)

    assert model(windows).shape == windows.shape


def test_cnn_supports_configured_window_lengths():
    model = CNNAutoencoder(n_features=3, window_length=19)
    windows = torch.randn(2, 19, 3)

    assert model(windows).shape == windows.shape


def test_cnn_rejects_unexpected_window_shape():
    model = CNNAutoencoder(n_features=3)

    with pytest.raises(ValueError, match="Expected windows"):
        model(torch.randn(2, 19, 3))


def test_cnn_works_with_shared_final_minute_scoring():
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(4, 20, 3)).astype(np.float32)

    scores, errors = score_autoencoder(CNNAutoencoder(n_features=3), windows)

    assert scores.shape == (4,)
    assert errors.shape == (4, 3)
    np.testing.assert_allclose(scores, errors.mean(axis=1))
