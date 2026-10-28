"""Run the windowed MLP experiments: a tune-loss sweep, then three seeds.

Usage:
    python scripts/run_windowed_mlp.py --sweep
    python scripts/run_windowed_mlp.py --latent-dim 16 --smoke
    python scripts/run_windowed_mlp.py --latent-dim 16
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from ugr16ad.data import load_dataset
from ugr16ad.evaluate import (
    fit_threshold,
    per_category_recall,
    ranking_metrics,
    threshold_metrics,
)
from ugr16ad.models.windowed_mlp import WindowedMLPAutoencoder, embed_windows
from ugr16ad.paths import RESULTS_DIR
from ugr16ad.preprocess import Preprocessor
from ugr16ad.results import run_dir, save_run
from ugr16ad.synthetic import inject_anomalies
from ugr16ad.train import score_autoencoder, set_seed, train_autoencoder
from ugr16ad.windows import make_windows

MODEL_NAME = "windowed_mlp"
WINDOW_LENGTH = 20
HIDDEN_DIM = 128
SEEDS = (0, 1, 2)
LATENT_DIMS = (8, 16, 32)
# Selection rule, fixed before any test result is seen: the smallest latent_dim
# whose best tune loss is within this fraction of the lowest.
TOLERANCE = 0.05
# Placeholders: the team plan leaves these for Phone to propose. Confirm before the final run.
SYNTHETIC = {"rate": 0.02, "magnitude": 5.0, "n_features": 10, "seed": 0}


def prepare() -> tuple[Preprocessor, dict, dict]:
    """Load, preprocess (fit on the fit part only) and window every part.

    Returns ``(preprocessor, parts, data)``. ``data[name]`` has ``x`` (windows,
    shape ``(n, WINDOW_LENGTH, n_features)``), ``timestamps`` (each window's final
    minute), ``labels`` (the 7 attack counts of that minute) and ``y`` (any > 0).
    """
    parts = load_dataset()
    prep = Preprocessor().fit(parts["fit"].X)
    data = {}
    for name, part in parts.items():
        windows, positions = make_windows(
            prep.transform(part.X), part.segment.to_numpy(), WINDOW_LENGTH
        )
        data[name] = {
            "x": windows,
            "timestamps": part.X.index[positions],
            "labels": part.labels.iloc[positions],
            "y": part.y.to_numpy()[positions],
        }
        n_attack = int(data[name]["y"].sum())
        print(f"{name}: {len(part.X)} rows -> {len(windows)} windows, {n_attack} attack")
    print(f"features after dropping constant columns: {len(prep.feature_names_)}")
    return prep, parts, data


def sweep(data: dict) -> None:
    """Seed 0, tune loss only. This never scores calib or test."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_features = data["fit"]["x"].shape[-1]
    rows = []
    for latent_dim in LATENT_DIMS:
        set_seed(0)
        model = WindowedMLPAutoencoder(WINDOW_LENGTH, n_features, HIDDEN_DIM, latent_dim)
        result = train_autoencoder(model, data["fit"]["x"], data["tune"]["x"], seed=0)
        curve = RESULTS_DIR / f"{MODEL_NAME}_loss_latent{latent_dim}.csv"
        result.history.to_csv(curve, index=False)
        rows.append(
            {
                "latent_dim": latent_dim,
                "best_tune_loss": result.history["tune_loss"].min(),
                "best_epoch": result.best_epoch,
                "seconds": round(result.seconds, 1),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / f"{MODEL_NAME}_sweep.csv", index=False)
    print(table.to_string(index=False))
    best = table["best_tune_loss"].min()
    within = table[table["best_tune_loss"] <= best * (1 + TOLERANCE)]
    print(f"\nSelection rule picks latent_dim={int(within['latent_dim'].min())}")


def synthetic_check(model, prep: Preprocessor, parts: dict, threshold: float) -> dict:
    """Do injected anomalies in the tune block score higher? Uses fit-part statistics only.

    Windows that merely contain an injected earlier minute count as negatives, so
    the false alert rate here is slightly pessimistic.
    """
    tune = parts["tune"]
    corrupted, mask = inject_anomalies(prep.transform(tune.X), **SYNTHETIC)
    windows, positions = make_windows(corrupted, tune.segment.to_numpy(), WINDOW_LENGTH)
    scores, _ = score_autoencoder(model, windows)
    injected = mask[positions]
    return {
        "settings": SYNTHETIC,
        "n_injected": int(injected.sum()),
        "median_score_injected": float(np.median(scores[injected])),
        "median_score_clean": float(np.median(scores[~injected])),
        "detection_rate_at_threshold": float(np.mean(scores[injected] > threshold)),
        "false_alert_rate_at_threshold": float(np.mean(scores[~injected] > threshold)),
        **ranking_metrics(injected, scores),
    }


def run_seed(prep: Preprocessor, parts: dict, data: dict, seed: int, latent_dim: int) -> None:
    n_features = data["fit"]["x"].shape[-1]
    set_seed(seed)  # before building the model, so initial weights are seeded too
    model = WindowedMLPAutoencoder(WINDOW_LENGTH, n_features, HIDDEN_DIM, latent_dim)
    result = train_autoencoder(model, data["fit"]["x"], data["tune"]["x"], seed=seed)

    calib, test = data["calib"], data["test"]
    started = time.perf_counter()
    calib_scores, calib_errors = score_autoencoder(result.model, calib["x"])
    test_scores, test_errors = score_autoencoder(result.model, test["x"])
    inference_seconds = time.perf_counter() - started

    threshold = fit_threshold(calib_scores)  # frozen from here on
    alerts = test_scores > threshold
    calib_alert_rate = float(np.mean(calib_scores > threshold))

    meta = {
        "model": MODEL_NAME,
        "seed": seed,
        "threshold": threshold,
        "window_length": WINDOW_LENGTH,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": latent_dim,
        "best_epoch": result.best_epoch,
        "train_seconds": result.seconds,
        "inference_seconds": inference_seconds,
        "n_parameters": sum(p.numel() for p in result.model.parameters()),
        "calib_alert_rate": calib_alert_rate,
        "threshold_metrics": threshold_metrics(test["y"], test_scores, threshold),
        "ranking_metrics": ranking_metrics(test["y"], test_scores),
        "per_category": per_category_recall(test["labels"], alerts).to_dict(orient="index"),
        "synthetic_check": synthetic_check(result.model, prep, parts, threshold),
    }
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
        run_dir(MODEL_NAME, seed),
        scores=scores,
        meta=meta,
        latents=np.concatenate(
            [embed_windows(result.model, calib["x"]), embed_windows(result.model, test["x"])]
        ),
        feature_errors=np.concatenate([calib_errors, test_errors]),
    )
    print(f"\nseed {seed}: calibration alert rate {calib_alert_rate:.4f}")
    print(f"  test: {meta['threshold_metrics']}")
    print(f"  synthetic: {meta['synthetic_check']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="tune-loss sweep; no test scoring")
    parser.add_argument("--latent-dim", type=int, help="the setting chosen by the rule")
    parser.add_argument("--smoke", action="store_true", help="run seed 0 only")
    args = parser.parse_args()

    prep, parts, data = prepare()
    if args.sweep:
        sweep(data)
    else:
        if not args.latent_dim:
            parser.error("run --sweep first, then pass --latent-dim")
        for seed in SEEDS[:1] if args.smoke else SEEDS:
            run_seed(prep, parts, data, seed, args.latent_dim)
