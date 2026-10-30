import numpy as np
import pytest
import torch

from ugr16ad.models.cnn import CNNAutoencoder
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


def test_cnn_final_minute_depends_on_every_minute_of_the_window():
    torch.manual_seed(0)
    model = CNNAutoencoder(n_features=3)
    windows = torch.randn(1, 20, 3, requires_grad=True)

    model(windows)[0, -1].sum().backward()

    assert (windows.grad[0].abs().sum(dim=1) > 0).all()


def test_cnn_bottleneck_is_one_vector_per_window():
    model = CNNAutoencoder(n_features=3, latent_dim=16)

    assert model.encode(torch.randn(4, 20, 3)).shape == (4, 16)
