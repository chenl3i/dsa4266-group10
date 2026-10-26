import numpy as np
import pytest
import torch
from torch import nn

from ugr16ad.models.mlp import MLPAutoencoder
from ugr16ad.train import score_autoencoder, set_seed, train_autoencoder


def test_shapes_and_gradients():
    model = MLPAutoencoder(6, hidden_dims=(5, 4), latent_dim=2)
    x = torch.randn(7, 6)
    reconstruction = model(x)
    assert reconstruction.shape == x.shape
    assert model.encoder(x).shape == (7, 2)
    (reconstruction - x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_output_can_reconstruct_negative_scaled_features():
    model = MLPAutoencoder(4, hidden_dims=(3,), latent_dim=2)
    assert isinstance(model.decoder[-1], nn.Linear)
    with torch.no_grad():
        model.decoder[-1].weight.zero_()
        model.decoder[-1].bias.fill_(-2)
    torch.testing.assert_close(model(torch.zeros(2, 4)), torch.full((2, 4), -2.0))


@pytest.mark.parametrize("shape", [(3, 2, 4), (3, 5), (4,)])
def test_rejects_windows_or_wrong_features(shape):
    with pytest.raises(ValueError):
        MLPAutoencoder(4, latent_dim=2)(torch.zeros(shape))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 0},
        {"input_dim": 4, "latent_dim": 4},
        {"input_dim": 4, "latent_dim": 2, "hidden_dims": (0,)},
    ],
)
def test_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        MLPAutoencoder(**kwargs)


def test_shared_training_and_scoring_on_unseen_minutes():
    rng = np.random.default_rng(8)
    x = (rng.normal(size=(160, 2)) @ rng.normal(size=(2, 4))).astype(np.float32)
    fit, tune = x[:120], x[120:]
    set_seed(8)
    model = MLPAutoencoder(4, hidden_dims=(12,), latent_dim=2)
    before = score_autoencoder(model, tune)[0].mean()
    result = train_autoencoder(
        model, fit, tune, epochs=40, batch_size=24, lr=0.01, patience=8, seed=8
    )
    scores, errors = score_autoencoder(result.model, tune)
    assert scores.mean() < before
    assert errors.shape == tune.shape
    np.testing.assert_allclose(scores, errors.mean(axis=1))
    np.testing.assert_allclose(scores.mean(), result.history.tune_loss.min(), rtol=1e-5)
