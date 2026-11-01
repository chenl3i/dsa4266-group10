"""Run the windowed MLP experiments: a tune-loss sweep, then three seeds.

Usage:
    python scripts/run_windowed_mlp.py --sweep
    python scripts/run_windowed_mlp.py --sweep --lr 3e-4
    python scripts/run_windowed_mlp.py --latent-dim 16 --smoke
    python scripts/run_windowed_mlp.py --latent-dim 16 --seed 1 2

All models use the shared protocol: up to 50 epochs, early-stopping patience 5.
Convergence check only (early stopping off, saved separately, not a main result):
    python scripts/run_windowed_mlp.py --latent-dim 16 --epochs 200 --patience 200 --tag converged
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ugr16ad.data import load_dataset
from ugr16ad.evaluate import (
    fit_threshold,
    per_category_recall,
    ranking_metrics,
    threshold_metrics,
)
from ugr16ad.models.windowed_mlp import WindowedMLPAutoencoder
from ugr16ad.paths import RESULTS_DIR
from ugr16ad.preprocess import Preprocessor
from ugr16ad.results import run_dir, save_run
from ugr16ad.synthetic import sanity_check
from ugr16ad.train import (
    encode_autoencoder,
    score_autoencoder,
    set_seed,
    train_autoencoder,
)
from ugr16ad.windows import make_windows

MODEL_NAME = "windowed_mlp"
WINDOW_LENGTH = 20  # agreed by the group
SEEDS = (0, 1, 2)
LATENT_DIMS = (8, 16, 32)
# Selection rule, fixed before any test result was seen: the smallest latent_dim
# whose best tune loss is within this fraction of the lowest.
TOLERANCE = 0.05
# Regularisation, a response to review: the first version (hidden 128, no dropout,
# no noise, lr 1e-3) had its lowest tune loss at epoch 1 for latent 8 and 16. These
# are fixed together and judged on tune loss only, never on test results.
HIDDEN_DIM = 64
DROPOUT = 0.2
NOISE_STD = 0.1  # in fit-part standard deviations, applied during training only
LR = 1e-4  # default; override with --lr
# The shared training protocol for every model (train_autoencoder's defaults). Raising
# both is only for the convergence check: with patience equal to epochs, early stopping
# never fires. The best-tune-loss weights are still the ones restored and scored.
EPOCHS = 50
PATIENCE = 5


def prepare() -> tuple[dict, dict]:
    """Load and preprocess every part. Windows are built later, only where needed.

    Returns ``(parts, features)``. ``features[name]`` is the 2-D standardised
    array of that part (small next to its windows, which are 20 times larger).
    The preprocessor is fit on the fit part only.
    """
    parts = load_dataset()
    prep = Preprocessor().fit(parts["fit"].X)
    features = {name: prep.transform(part.X) for name, part in parts.items()}
    print(f"features after dropping constant columns: {len(prep.feature_names_)}")
    return parts, features


def window_part(parts: dict, features: dict, name: str) -> dict:
    """Window one part: ``x``, final-minute ``timestamps``, ``labels`` and ``y``."""
    part = parts[name]
    windows, positions = make_windows(features[name], part.segment.to_numpy(), WINDOW_LENGTH)
    y = part.y.to_numpy()[positions]
    print(f"{name}: {len(part.X)} rows -> {len(windows)} windows, {int(y.sum())} attack")
    return {
        "x": windows,
        "timestamps": part.X.index[positions],
        "labels": part.labels.iloc[positions],
        "y": y,
    }


def build_and_train(
    fit: dict, tune: dict, latent_dim: int, seed: int, lr: float, epochs: int, patience: int
):
    """One place that builds and trains the model, so the sweep and final runs agree."""
    set_seed(seed)  # before building the model, so initial weights are seeded too
    model = WindowedMLPAutoencoder(
        fit["x"].shape[-1],
        WINDOW_LENGTH,
        HIDDEN_DIM,
        latent_dim,
        dropout=DROPOUT,
        noise_std=NOISE_STD,
    )
    # train_autoencoder prints nothing per epoch, so say that a long, quiet step began.
    print(
        f"training latent_dim={latent_dim} seed={seed} lr={lr:g} "
        f"epochs={epochs} patience={patience} (quiet until it finishes)..."
    )
    return train_autoencoder(
        model, fit["x"], tune["x"], lr=lr, seed=seed, epochs=epochs, patience=patience
    )


def sweep(parts: dict, features: dict, lr: float, epochs: int, patience: int, name: str) -> None:
    """Seed 0, tune loss only. This never windows or scores calib or test."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fit = window_part(parts, features, "fit")
    tune = window_part(parts, features, "tune")
    rows = []
    for latent_dim in LATENT_DIMS:
        result = build_and_train(fit, tune, latent_dim, 0, lr, epochs, patience)
        curve = RESULTS_DIR / f"{name}_loss_latent{latent_dim}_lr{lr:g}.csv"
        result.history.to_csv(curve, index=False)
        rows.append(
            {
                "latent_dim": latent_dim,
                "lr": lr,
                "best_tune_loss": result.history["tune_loss"].min(),
                "best_epoch": result.best_epoch,
                "epochs_run": len(result.history),
                "seconds": round(result.seconds, 1),
            }
        )
        print(rows[-1])
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / f"{name}_sweep_lr{lr:g}.csv", index=False)
    print(table.to_string(index=False))
    best = table["best_tune_loss"].min()
    within = table[table["best_tune_loss"] <= best * (1 + TOLERANCE)]
    print(f"\nSelection rule picks latent_dim={int(within['latent_dim'].min())}")


def run_seed(
    parts: dict,
    features: dict,
    fit: dict,
    tune: dict,
    seed: int,
    latent_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    name: str,
) -> None:
    result = build_and_train(fit, tune, latent_dim, seed, lr, epochs, patience)

    # Calib and test are windowed only now, so they are not in memory during training,
    # and they are freed when this function returns.
    calib = window_part(parts, features, "calib")
    test = window_part(parts, features, "test")
    started = time.perf_counter()
    calib_scores, calib_errors = score_autoencoder(result.model, calib["x"])
    test_scores, test_errors = score_autoencoder(result.model, test["x"])
    inference_seconds = time.perf_counter() - started

    threshold = fit_threshold(calib_scores)  # frozen from here on
    alerts = test_scores > threshold
    calib_alert_rate = float(np.mean(calib_scores > threshold))

    def score_calib(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        windows, positions = make_windows(X, parts["calib"].segment.to_numpy(), WINDOW_LENGTH)
        return score_autoencoder(result.model, windows)[0], positions

    meta = {
        "model": name,
        "seed": seed,
        "threshold": threshold,
        "window_length": WINDOW_LENGTH,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": latent_dim,
        "dropout": DROPOUT,
        "noise_std": NOISE_STD,
        "lr": lr,
        "epochs": epochs,
        "patience": patience,
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "train_seconds": result.seconds,
        "inference_seconds": inference_seconds,
        "n_parameters": sum(p.numel() for p in result.model.parameters()),
        "calib_alert_rate": calib_alert_rate,
        "test_metrics": {
            **threshold_metrics(test["y"], test_scores, threshold),
            **ranking_metrics(test["y"], test_scores),
            # The usual full ROC-AUC, next to the stricter partial AUC over FPR 0-1%.
            "roc_auc": float(roc_auc_score(test["y"], test_scores)),
        },
        "per_category_recall": per_category_recall(test["labels"], alerts)
        .reset_index(names="category")
        .to_dict("records"),
    }
    # The agreed check: calibration block, rate and grid from synthetic.py.
    # features["calib"] is the same array as prep.transform(parts["calib"].X).
    meta["sanity_check"] = sanity_check(features["calib"], score_calib)

    scores = pd.concat(
        [
            pd.DataFrame(
                {"timestamp": calib["timestamps"], "part": "calib", "score": calib_scores}
            ),
            pd.DataFrame({"timestamp": test["timestamps"], "part": "test", "score": test_scores}),
        ],
        ignore_index=True,
    )
    save_run(
        run_dir(name, seed),
        scores=scores,
        meta=meta,
        latents=np.concatenate(
            [encode_autoencoder(result.model, part["x"]) for part in (calib, test)]
        ),
        feature_errors=np.concatenate([calib_errors, test_errors]),
    )
    # The loss curve of every reported seed, next to its scores.
    result.history.to_csv(run_dir(name, seed) / "history.csv", index=False)
    print(f"\nseed {seed}: calibration alert rate {calib_alert_rate:.4f}")
    print(f"  best epoch {result.best_epoch} of {len(result.history)}")
    print(f"  test: {meta['test_metrics']}")
    print(f"  sanity check: {meta['sanity_check']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="tune-loss sweep; no test scoring")
    parser.add_argument("--latent-dim", type=int, help="the setting chosen by the rule")
    parser.add_argument("--lr", type=float, default=LR, help=f"learning rate (default {LR:g})")
    parser.add_argument("--smoke", action="store_true", help="run seed 0 only")
    parser.add_argument("--seed", type=int, nargs="+", help="seeds to run, e.g. --seed 1 2")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="maximum epochs")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="early-stopping patience")
    parser.add_argument(
        "--tag", help="save as windowed_mlp_<tag> so a variant never overwrites the main runs"
    )
    args = parser.parse_args()
    name = f"{MODEL_NAME}_{args.tag}" if args.tag else MODEL_NAME

    parts, features = prepare()
    if args.sweep:
        sweep(parts, features, args.lr, args.epochs, args.patience, name)
    else:
        if not args.latent_dim:
            parser.error("run --sweep first, then pass --latent-dim")
        seeds = args.seed or (SEEDS[:1] if args.smoke else SEEDS)
        fit = window_part(parts, features, "fit")
        tune = window_part(parts, features, "tune")
        for seed in seeds:
            run_seed(
                parts,
                features,
                fit,
                tune,
                seed,
                args.latent_dim,
                args.lr,
                args.epochs,
                args.patience,
                name,
            )
