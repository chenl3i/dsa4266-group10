"""Run the windowed MLP experiments: a tune-loss sweep, then three seeds.

Usage:
    python scripts/run_windowed_mlp.py --sweep
    python scripts/run_windowed_mlp.py --sweep --lr 3e-4
    python scripts/run_windowed_mlp.py --latent-dim 16 --smoke
    python scripts/run_windowed_mlp.py --latent-dim 16 --seed 1 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
from ugr16ad.selection import check_blind_root, tune_sanity
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
    fit: dict,
    tune: dict,
    latent_dim: int,
    seed: int,
    lr: float,
    patience: int = 5,
    hidden_dim: int = HIDDEN_DIM,
    dropout: float = DROPOUT,
    noise_std: float = NOISE_STD,
):
    """One place that builds and trains the model, so the sweep and final runs agree."""
    set_seed(seed)  # before building the model, so initial weights are seeded too
    model = WindowedMLPAutoencoder(
        fit["x"].shape[-1],
        WINDOW_LENGTH,
        hidden_dim,
        latent_dim,
        dropout=dropout,
        noise_std=noise_std,
    )
    # train_autoencoder prints nothing per epoch, so say that a long, quiet step began.
    print(f"training latent_dim={latent_dim} seed={seed} lr={lr:g} (quiet until it finishes)...")
    return train_autoencoder(model, fit["x"], tune["x"], lr=lr, seed=seed, patience=patience)


def sweep(parts: dict, features: dict, lr: float) -> None:
    """Seed 0, tune loss only. This never windows or scores calib or test."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fit = window_part(parts, features, "fit")
    tune = window_part(parts, features, "tune")
    rows = []
    for latent_dim in LATENT_DIMS:
        result = build_and_train(fit, tune, latent_dim, seed=0, lr=lr)
        curve = RESULTS_DIR / f"{MODEL_NAME}_loss_latent{latent_dim}_lr{lr:g}.csv"
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
    table.to_csv(RESULTS_DIR / f"{MODEL_NAME}_sweep_lr{lr:g}.csv", index=False)
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
    patience: int = 5,
    hidden_dim: int = HIDDEN_DIM,
    dropout: float = DROPOUT,
    noise_std: float = NOISE_STD,
    out_root: Path = RESULTS_DIR,
    blind: bool = False,
) -> None:
    """Train one seed and save it.

    ``blind`` is for the sweep: test is never windowed or scored, so no config can
    be picked on it, and the run must not land in ``RESULTS_DIR`` where the
    comparison table would read it as a finished run.
    """
    if blind:
        check_blind_root(out_root)
    result = build_and_train(
        fit, tune, latent_dim, seed, lr, patience, hidden_dim, dropout, noise_std
    )

    # Calib and test are windowed only now, so they are not in memory during training,
    # and they are freed when this function returns.
    calib = window_part(parts, features, "calib")
    test = None if blind else window_part(parts, features, "test")
    started = time.perf_counter()
    calib_scores, calib_errors = score_autoencoder(result.model, calib["x"])
    if not blind:
        test_scores, test_errors = score_autoencoder(result.model, test["x"])
    inference_seconds = time.perf_counter() - started
    # Outside the timer: inference_seconds is calib + test scoring only.
    tune_scores = score_autoencoder(result.model, tune["x"])[0]

    threshold = fit_threshold(calib_scores)  # frozen from here on
    tune_threshold = fit_threshold(tune_scores)
    calib_alert_rate = float(np.mean(calib_scores > threshold))

    def scorer(name: str):
        def score_rows(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            windows, positions = make_windows(X, parts[name].segment.to_numpy(), WINDOW_LENGTH)
            return score_autoencoder(result.model, windows)[0], positions

        return score_rows

    meta = {
        "model": MODEL_NAME,
        "seed": seed,
        "threshold": threshold,
        "window_length": WINDOW_LENGTH,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "dropout": dropout,
        "noise_std": noise_std,
        "lr": lr,
        "patience": patience,
        # Parallel sweep workers share the CPU, so their train_seconds are not comparable.
        "threads": torch.get_num_threads(),
        "blind": blind,
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "best_tune_loss": float(result.history["tune_loss"].min()),
        "train_seconds": result.seconds,
        "n_parameters": sum(p.numel() for p in result.model.parameters()),
        "calib_alert_rate": calib_alert_rate,
        "tune_threshold": tune_threshold,
        "tune_sanity": tune_sanity(features["tune"], scorer("tune"), tune_threshold),
    }
    if blind:
        frames = [
            pd.DataFrame({"timestamp": part["timestamps"], "part": name, "score": scores})
            for name, part, scores in (
                ("tune", tune, tune_scores),
                ("calib", calib, calib_scores),
            )
        ]
        latents = errors = None
    else:
        alerts = test_scores > threshold
        meta |= {
            "inference_seconds": inference_seconds,
            "test_metrics": {
                **threshold_metrics(test["y"], test_scores, threshold),
                **ranking_metrics(test["y"], test_scores),
            },
            "per_category_recall": per_category_recall(test["labels"], alerts)
            .reset_index(names="category")
            .to_dict("records"),
            # The agreed check: calibration block, rate and grid from synthetic.py.
            # features["calib"] is the same array as prep.transform(parts["calib"].X).
            "sanity_check": sanity_check(features["calib"], scorer("calib")),
        }
        frames = [
            pd.DataFrame(
                {"timestamp": calib["timestamps"], "part": "calib", "score": calib_scores}
            ),
            pd.DataFrame({"timestamp": test["timestamps"], "part": "test", "score": test_scores}),
        ]
        latents = np.concatenate(
            [encode_autoencoder(result.model, part["x"]) for part in (calib, test)]
        )
        errors = np.concatenate([calib_errors, test_errors])
    directory = run_dir(MODEL_NAME, seed, root=out_root)
    save_run(
        directory,
        scores=pd.concat(frames, ignore_index=True),
        meta=meta,
        latents=latents,
        feature_errors=errors,
    )
    # The loss curve of every reported seed, next to its scores.
    result.history.to_csv(directory / "history.csv", index=False)
    print(f"\nseed {seed}: calibration alert rate {calib_alert_rate:.4f}")
    print(f"  best epoch {result.best_epoch} of {len(result.history)}")
    if not blind:
        print(f"  test: {meta['test_metrics']}")
        print(f"  sanity check: {meta['sanity_check']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="tune-loss sweep; no test scoring")
    parser.add_argument("--latent-dim", type=int, help="the setting chosen by the rule")
    parser.add_argument("--lr", type=float, default=LR, help=f"learning rate (default {LR:g})")
    parser.add_argument("--smoke", action="store_true", help="run seed 0 only")
    parser.add_argument("--seed", type=int, nargs="+", help="seeds to run, e.g. --seed 1 2")
    parser.add_argument(
        "--patience", type=int, default=5, help="early stopping; 50 trains every epoch"
    )
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--noise-std", type=float, default=NOISE_STD)
    parser.add_argument("--threads", type=int, help="torch CPU threads (default: torch's)")
    parser.add_argument("--out-root", type=Path, default=RESULTS_DIR)
    parser.add_argument("--blind", action="store_true", help="skip test; for the sweep")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    parts, features = prepare()
    if args.sweep:
        sweep(parts, features, args.lr)
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
                args.patience,
                args.hidden_dim,
                args.dropout,
                args.noise_std,
                args.out_root,
                args.blind,
            )
