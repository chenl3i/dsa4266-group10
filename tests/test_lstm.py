import numpy as np
import pytest
import torch
from torch import nn

from ugr16ad.models.lstm import LSTMAutoencoder
from ugr16ad.train import score_autoencoder, set_seed, train_autoencoder

W, F = 5, 6


def make_model(seed=0, cell="lstm"):
    set_seed(seed)
    return LSTMAutoencoder(n_features=F, hidden_dim=16, latent_dim=4, cell=cell)


def low_rank_windows(n=128):
    rng = np.random.default_rng(0)
    flat = rng.normal(size=(n, 3)) @ rng.normal(size=(3, W * F))
    return flat.reshape(n, W, F).astype(np.float32)


@pytest.mark.parametrize("cell", ["lstm", "gru"])
@pytest.mark.parametrize("length", [1, 5, 20])
def test_output_shape_matches_input_shape(cell, length):
    x = torch.randn(3, length, F)
    assert make_model(cell=cell)(x).shape == x.shape


def test_encode_returns_the_bottleneck():
    assert make_model().encode(torch.randn(8, W, F)).shape == (8, 4)


def test_gru_cell_uses_gru():
    modules = list(make_model(cell="gru").modules())
    assert any(isinstance(m, nn.GRU) for m in modules)
    assert not any(isinstance(m, nn.LSTM) for m in modules)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_features": 0}, {"hidden_dim": 0}, {"latent_dim": 0}, {"cell": "rnn"}],
)
def test_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        LSTMAutoencoder(**{"n_features": F, **kwargs})


def test_training_lowers_the_loss():
    d = low_rank_windows()
    result = train_autoencoder(
        make_model(), d[:96], d[96:], epochs=40, batch_size=16, lr=1e-2, patience=100
    )
    assert result.history["train_loss"].iloc[-1] < result.history["train_loss"].iloc[0]


def test_scores_use_the_final_minute_only():
    d = low_rank_windows()
    model = train_autoencoder(make_model(), d[:96], d[96:], epochs=3).model
    _, feature_errors = score_autoencoder(model, d)
    with torch.no_grad():
        recon = model.eval()(torch.tensor(d)).numpy()
    np.testing.assert_allclose(feature_errors, (recon[:, -1] - d[:, -1]) ** 2, rtol=1e-5, atol=1e-6)


def test_score_does_not_depend_on_batch():
    d = low_rank_windows()
    model = make_model()
    one, _ = score_autoencoder(model, d, batch_size=1)
    many, _ = score_autoencoder(model, d, batch_size=1024)
    np.testing.assert_allclose(one, many, atol=1e-6)


def test_same_seed_reproduces_the_scores():
    d = low_rank_windows()
    scores = []
    for _ in range(2):
        model = train_autoencoder(make_model(seed=1), d[:96], d[96:], epochs=3).model
        scores.append(score_autoencoder(model, d)[0])
    np.testing.assert_array_equal(scores[0], scores[1])
