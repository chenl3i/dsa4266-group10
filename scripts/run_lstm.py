"""Train the LSTM/GRU window autoencoder and save it in the standard results format.

Run: ./.venv/Scripts/python.exe scripts/run_lstm.py --seed 0 1 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ugr16ad.data import Part, load_dataset
from ugr16ad.evaluate import (
    fit_threshold,
    per_category_recall,
    ranking_metrics,
    threshold_metrics,
)
from ugr16ad.logging_utils import get_logger, setup_logging
from ugr16ad.models.lstm import LSTMAutoencoder
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

log = get_logger(__name__)

PART_ORDER = ("fit", "tune", "calib", "test")


def run(
    parts: dict[str, Part],
    *,
    seed: int,
    out_root: Path,
    window_length: int = 20,
    cell: str = "lstm",
    hidden_dim: int = 64,
    latent_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
) -> Path:
    """Run the shared pipeline around ``LSTMAutoencoder`` and return the run directory.

    The folder is named after ``cell`` so LSTM and GRU results never overwrite
    each other. Threshold and test metrics follow the frozen-threshold rules:
    the threshold comes from calib only and is applied unchanged to test.
    """
    preprocessor = Preprocessor().fit(parts["fit"].X)
    windows, positions = {}, {}
    for name in PART_ORDER:
        part = parts[name]
        transformed = preprocessor.transform(part.X)
        if name == "calib":
            calib_X = transformed  # kept clean for the synthetic sanity check
        windows[name], positions[name] = make_windows(
            transformed, part.segment.to_numpy(), window_length
        )
        if len(windows[name]) == 0:
            raise ValueError(f"no {window_length}-minute windows in {name}")

    set_seed(seed)
    model = LSTMAutoencoder(
        windows["fit"].shape[2], hidden_dim=hidden_dim, latent_dim=latent_dim, cell=cell
    )
    result = train_autoencoder(
        model,
        windows["fit"],
        windows["tune"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        seed=seed,
        device=device,
    )
    model = result.model
    log.info("%s seed %d: best epoch %d in %.1fs", cell, seed, result.best_epoch, result.seconds)

    # inference_seconds times calib + test scoring only, the same span in every script.
    started = time.perf_counter()
    scored = {
        name: score_autoencoder(model, windows[name], device=device) for name in ("calib", "test")
    }
    inference_seconds = time.perf_counter() - started
    for name in ("fit", "tune"):
        scored[name] = score_autoencoder(model, windows[name], device=device)

    frames, latents, errors, scores_by_part = [], [], [], {}
    for name in PART_ORDER:
        scores, feature_errors = scored[name]
        scores_by_part[name] = scores
        errors.append(feature_errors)
        latents.append(
            encode_autoencoder(model, windows[name], batch_size=batch_size, device=device)
        )
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": parts[name].X.index[positions[name]],
                    "part": name,
                    "score": scores,
                }
            )
        )

    threshold = fit_threshold(scores_by_part["calib"])

    def score_calib(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, pos = make_windows(X, parts["calib"].segment.to_numpy(), window_length)
        return score_autoencoder(model, w, device=device)[0], pos

    test = parts["test"]
    y_test = test.y.to_numpy()[positions["test"]]
    test_scores = scores_by_part["test"]
    alerts = test_scores > threshold
    recall = per_category_recall(test.labels.iloc[positions["test"]], alerts)

    meta = {
        "model": cell,
        "seed": seed,
        "threshold": threshold,
        "window_length": window_length,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "n_features": windows["fit"].shape[2],
        "epochs": epochs,
        "patience": patience,
        "device": device,
        "lr": lr,
        "batch_size": batch_size,
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "train_seconds": result.seconds,
        "inference_seconds": inference_seconds,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        # Sanity check that the frozen threshold alerts on about 1% of calib.
        "calib_alert_rate": float((scores_by_part["calib"] > threshold).mean()),
        "test_metrics": {
            **threshold_metrics(y_test, test_scores, threshold),
            **ranking_metrics(y_test, test_scores),
        },
        "per_category_recall": recall.reset_index(names="category").to_dict("records"),
        "sanity_check": sanity_check(calib_X, score_calib),
    }
    directory = run_dir(cell, seed, root=out_root)
    save_run(
        directory,
        scores=pd.concat(frames, ignore_index=True),
        meta=meta,
        latents=np.concatenate(latents),
        feature_errors=np.concatenate(errors),
    )
    # save_run knows nothing about the loss curve, so it is written alongside.
    result.history.to_csv(directory / "history.csv", index=False)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--window-length", type=int, default=20)
    parser.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    setup_logging()
    parts = load_dataset()
    for seed in args.seed:
        directory = run(
            parts,
            seed=seed,
            out_root=RESULTS_DIR,
            window_length=args.window_length,
            cell=args.cell,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device=args.device,
        )
        log.info("wrote %s", directory)


if __name__ == "__main__":
    main()
