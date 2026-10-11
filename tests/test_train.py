import numpy as np
import torch
from torch import nn

from stubs import todo
from ugr16ad.train import score_autoencoder, set_seed, train_autoencoder


class TinyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 4))

    def forward(self, x):
        return self.net(x)


class Zeros(nn.Module):
    """Reconstructs everything as 0, so the error equals the input squared."""

    def forward(self, x):
        return torch.zeros_like(x)


class Bias(nn.Module):
    """Ignores its input and reconstructs a learned constant per feature.

    Starts at 0, so its loss is exactly the mean of the squared inputs until Adam
    moves it; that makes every reported loss predictable.
    """

    def __init__(self, features=4):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        return self.b.expand_as(x)


class Dropy(nn.Module):
    """Dropout autoencoder: scoring is only repeatable if the model is in eval mode."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 4))

    def forward(self, x):
        return self.net(x)


def _data():
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(200, 2)) @ rng.normal(size=(2, 4))).astype(np.float32)
    return x[:150], x[150:]


@todo
def test_training_lowers_loss_and_keeps_best_epoch():
    train, tune = _data()
    set_seed(0)
    result = train_autoencoder(TinyAE(), train, tune, epochs=20, batch_size=32, lr=1e-2, seed=0)
    h = result.history
    assert list(h.columns) == ["epoch", "train_loss", "tune_loss"]
    assert h["epoch"].iloc[0] == 1
    assert h["train_loss"].iloc[-1] < h["train_loss"].iloc[0]
    assert result.best_epoch == int(h.loc[h["tune_loss"].idxmin(), "epoch"])
    assert result.seconds > 0


@todo
def test_early_stopping_after_patience_epochs_without_improvement():
    train, tune = _data()
    set_seed(0)
    result = train_autoencoder(TinyAE(), train, tune, epochs=50, lr=0.0, patience=3, seed=0)
    assert len(result.history) == 4  # best at epoch 1, then 3 epochs with no improvement


@todo
def test_same_seed_same_history():
    train, tune = _data()
    runs = []
    for _ in range(2):
        set_seed(0)
        runs.append(train_autoencoder(TinyAE(), train, tune, epochs=5, seed=0).history)
    assert runs[0].equals(runs[1])


@todo
def test_score_single_minutes():
    x = np.array([[1, 2], [3, 0]], dtype=np.float32)
    scores, errors = score_autoencoder(Zeros(), x)
    np.testing.assert_allclose(errors, x**2)
    np.testing.assert_allclose(scores, [2.5, 4.5])


@todo
def test_score_windows_uses_final_minute_only():
    x = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    scores, errors = score_autoencoder(Zeros(), x)
    np.testing.assert_allclose(errors, x[:, -1, :] ** 2)
    np.testing.assert_allclose(scores, (x[:, -1, :] ** 2).mean(axis=1))


@todo
def test_best_epoch_weights_are_restored_not_last_epoch():
    # Train rows sit at 5 but tune rows at 0: every Adam step moves the constant
    # away from what the tune rows want, so epoch 1 is best and the last is worst.
    train = np.full((150, 4), 5.0, dtype=np.float32)
    tune = np.zeros((50, 4), dtype=np.float32)
    result = train_autoencoder(Bias(), train, tune, epochs=20, batch_size=150, lr=0.5, patience=2)
    h = result.history
    assert result.best_epoch == 1
    assert len(h) == 3
    assert h["tune_loss"].iloc[-1] > h["tune_loss"].iloc[0]
    scores, _ = score_autoencoder(result.model, tune)
    np.testing.assert_allclose(scores.mean(), h["tune_loss"].min(), rtol=1e-5)


@todo
def test_history_has_one_row_per_epoch_and_runs_all_epochs_while_improving():
    # Both parts sit at 5 and the constant starts at 0, so each step moves toward
    # the optimum and tune loss falls every epoch.
    data = np.full((100, 4), 5.0, dtype=np.float32)
    result = train_autoencoder(Bias(), data, data, epochs=5, batch_size=50, lr=0.1)
    h = result.history
    assert list(h["epoch"]) == [1, 2, 3, 4, 5]
    assert h["tune_loss"].is_monotonic_decreasing
    assert result.best_epoch == 5


@todo
def test_losses_are_mean_squared_error_over_all_elements():
    # lr=0 leaves the constant at 0, so the reconstruction error is the input squared.
    train, tune = _data()
    result = train_autoencoder(Bias(), train, tune, epochs=1, batch_size=50, lr=0.0)
    row = result.history.iloc[0]
    np.testing.assert_allclose(row["train_loss"], (train**2).mean(), rtol=1e-4)
    np.testing.assert_allclose(row["tune_loss"], (tune**2).mean(), rtol=1e-4)


@todo
def test_training_accepts_windows():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(60, 3, 4)).astype(np.float32)
    tune = rng.normal(size=(20, 3, 4)).astype(np.float32)
    result = train_autoencoder(Bias(), train, tune, epochs=1, batch_size=16, lr=0.0)
    np.testing.assert_allclose(result.history["tune_loss"].iloc[0], (tune**2).mean(), rtol=1e-4)


@todo
def test_different_seeds_shuffle_differently_and_inputs_are_untouched():
    train, tune = _data()
    train_before, tune_before = train.copy(), tune.copy()
    histories = []
    for seed in (1, 2):
        set_seed(0)  # same weights; only the mini-batch order differs
        histories.append(
            train_autoencoder(TinyAE(), train, tune, epochs=3, batch_size=16, seed=seed)
        )
    assert not histories[0].history.equals(histories[1].history)
    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(tune, tune_before)


@todo
def test_set_seed_makes_weight_initialisation_repeatable():
    set_seed(3)
    first = TinyAE().net[0].weight.detach().clone()
    set_seed(3)
    second = TinyAE().net[0].weight.detach().clone()
    assert torch.equal(first, second)
    set_seed(4)
    assert not torch.equal(first, TinyAE().net[0].weight)


@todo
def test_score_is_independent_of_batch_size():
    train, _ = _data()
    set_seed(0)
    model = TinyAE()
    one = score_autoencoder(model, train, batch_size=1)
    many = score_autoencoder(model, train, batch_size=1024)
    uneven = score_autoencoder(model, train, batch_size=64)
    for other in (many, uneven):
        np.testing.assert_allclose(one[0], other[0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(one[1], other[1], rtol=1e-5, atol=1e-6)
    assert one[0].shape == (150,)
    assert one[1].shape == (150, 4)


@todo
def test_scoring_is_deterministic_even_for_a_model_left_in_train_mode():
    train, tune = _data()
    set_seed(0)
    model = Dropy()
    first = score_autoencoder(model, tune)[0]
    model.train()
    second = score_autoencoder(model, tune)[0]
    np.testing.assert_allclose(first, second)
    trained = train_autoencoder(Dropy(), train, tune, epochs=2).model
    np.testing.assert_allclose(
        score_autoencoder(trained, tune)[0], score_autoencoder(trained, tune)[0]
    )


@todo
def test_score_windows_batches_match_unbatched():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3, 4)).astype(np.float32)
    set_seed(0)
    model = TinyAE()
    a = score_autoencoder(model, x, batch_size=3)
    b = score_autoencoder(model, x, batch_size=100)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-5, atol=1e-6)
    assert a[1].shape == (10, 4)
