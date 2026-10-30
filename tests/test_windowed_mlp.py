import numpy as np
import pytest
import torch

from ugr16ad.models.windowed_mlp import WindowedMLPAutoencoder, embed_windows
from ugr16ad.train import score_autoencoder, set_seed, train_autoencoder

W, F = 5, 6


def make_model(seed: int = 0) -> WindowedMLPAutoencoder:
    set_seed(seed)  # train_autoencoder seeds after the model exists, so seed init here
    return WindowedMLPAutoencoder(window_length=W, n_features=F, hidden_dim=32, latent_dim=4)


def low_rank_windows(n: int = 128) -> np.ndarray:
    """Windows with rank-3 structure, which a latent of 4 can reconstruct."""
    rng = np.random.default_rng(0)
    flat = rng.normal(size=(n, 3)) @ rng.normal(size=(3, W * F))
    return flat.reshape(n, W, F).astype(np.float32)


def test_output_shape_matches_input_shape():
    x = torch.randn(8, W, F)
    assert make_model()(x).shape == x.shape


def test_encode_returns_the_bottleneck():
    assert make_model().encode(torch.randn(8, W, F)).shape == (8, 4)


def test_rejects_non_positive_sizes():
    with pytest.raises(ValueError):
        WindowedMLPAutoencoder(window_length=0, n_features=F)


def test_training_lowers_the_loss():
    data = low_rank_windows()
    result = train_autoencoder(
        make_model(), data[:96], data[96:], epochs=60, batch_size=16, lr=1e-2, patience=100
    )
    assert result.history["train_loss"].iloc[-1] < result.history["train_loss"].iloc[0]
    assert list(result.history.columns) == ["epoch", "train_loss", "tune_loss"]


def test_scores_use_the_final_minute_only():
    data = low_rank_windows()
    result = train_autoencoder(make_model(), data[:96], data[96:], epochs=3, batch_size=16)
    scores, feature_errors = score_autoencoder(result.model, data)
    with torch.no_grad():
        recon = result.model(torch.tensor(data)).numpy()
    expected = (recon[:, -1, :] - data[:, -1, :]) ** 2
    assert scores.shape == (len(data),)
    assert feature_errors.shape == (len(data), F)
    np.testing.assert_allclose(feature_errors, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(scores, expected.mean(axis=1), rtol=1e-5, atol=1e-6)


def test_same_seed_reproduces_the_scores():
    data = low_rank_windows()
    runs = []
    for _ in range(2):
        result = train_autoencoder(
            make_model(seed=1), data[:96], data[96:], epochs=3, batch_size=16, seed=1
        )
        runs.append(score_autoencoder(result.model, data)[0])
    np.testing.assert_array_equal(runs[0], runs[1])


def test_embed_windows_is_row_aligned_with_the_scores():
    data = low_rank_windows()
    model = make_model()
    scores, _ = score_autoencoder(model, data)
    assert embed_windows(model, data).shape == (len(scores), 4)


def test_regularisers_are_off_by_default():
    model = WindowedMLPAutoencoder(window_length=W, n_features=F)
    model.train()
    x = torch.randn(4, W, F)
    assert torch.equal(model(x), model(x))


def test_noise_and_dropout_only_act_in_training_mode():
    torch.manual_seed(0)
    model = WindowedMLPAutoencoder(W, F, hidden_dim=32, latent_dim=4, dropout=0.5, noise_std=0.5)
    x = torch.randn(4, W, F)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    assert not torch.equal(model(x), model(x))


def test_rejects_invalid_dropout_and_noise():
    with pytest.raises(ValueError):
        WindowedMLPAutoencoder(W, F, dropout=1.0)
    with pytest.raises(ValueError):
        WindowedMLPAutoencoder(W, F, noise_std=-0.1)


def test_regularised_model_still_trains_against_the_clean_input():
    data = low_rank_windows()
    set_seed(0)
    model = WindowedMLPAutoencoder(W, F, hidden_dim=32, latent_dim=4, dropout=0.1, noise_std=0.1)
    result = train_autoencoder(
        model, data[:96], data[96:], epochs=80, batch_size=16, lr=1e-2, patience=100
    )
    assert result.history["tune_loss"].min() < result.history["tune_loss"].iloc[0]
