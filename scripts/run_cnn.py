"""Train the 1D-CNN window autoencoder and save it in the standard results format.

The CNN goes through exactly the same steps as the LSTM and windowed MLP (same
preprocessing, windows, training loop, frozen threshold and sanity check), so a
difference in the comparison table comes from the model, not the plumbing.

Run: ./.venv/Scripts/python.exe scripts/run_cnn.py --seed 0 1 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ugr16ad.data import Part, load_dataset
from ugr16ad.evaluate import (
    fit_threshold,
    per_category_recall,
    ranking_metrics,
    threshold_metrics,
)
from ugr16ad.logging_utils import get_logger, setup_logging
from ugr16ad.models.cnn import CNNAutoencoder
from ugr16ad.paths import RESULTS_DIR
from ugr16ad.preprocess import Preprocessor
from ugr16ad.results import run_dir, save_run
from ugr16ad.synthetic import sanity_check
from ugr16ad.train import score_autoencoder, set_seed, train_autoencoder
from ugr16ad.windows import make_windows

log = get_logger(__name__)

PART_ORDER = ("fit", "tune", "calib", "test")


def _encode(model: CNNAutoencoder, windows: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """Latent vectors, batched so the full fit part never sits on the device at once."""
    model.eval()
    out = []
    with torch.no_grad():
        for batch in torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)).split(
            batch_size
        ):
            out.append(model.encode(batch.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def run(
    parts: dict[str, Part],
    *,
    seed: int,
    out_root: Path,
    window_length: int = 20,
    hidden_channels: int = 32,
    latent_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
) -> Path:
    """Run the shared pipeline around ``CNNAutoencoder`` and return the run directory.

    The threshold comes from calib only and is applied unchanged to test.
    ``inference_seconds`` times calib + test scoring, the same span the windowed MLP
    times, so the speed column compares like with like.
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
    model = CNNAutoencoder(
        windows["fit"].shape[2],
        window_length,
        hidden_channels=hidden_channels,
        latent_dim=latent_dim,
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
    log.info("cnn seed %d: best epoch %d in %.1fs", seed, result.best_epoch, result.seconds)

    started = time.perf_counter()
    scored = {
        name: score_autoencoder(model, windows[name], device=device) for name in ("calib", "test")
    }
    inference_seconds = time.perf_counter() - started
    for name in ("fit", "tune"):
        scored[name] = score_autoencoder(model, windows[name], device=device)

    frames, latents, errors = [], [], []
    for name in PART_ORDER:
        scores, feature_errors = scored[name]
        errors.append(feature_errors)
        latents.append(_encode(model, windows[name], batch_size, device))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": parts[name].X.index[positions[name]],
                    "part": name,
                    "score": scores,
                }
            )
        )

    calib_scores = scored["calib"][0]
    threshold = fit_threshold(calib_scores)

    def score_calib(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, pos = make_windows(X, parts["calib"].segment.to_numpy(), window_length)
        return score_autoencoder(model, w, device=device)[0], pos

    test = parts["test"]
    y_test = test.y.to_numpy()[positions["test"]]
    test_scores = scored["test"][0]
    alerts = test_scores > threshold
    recall = per_category_recall(test.labels.iloc[positions["test"]], alerts)

    meta = {
        "model": "cnn",
        "seed": seed,
        "threshold": threshold,
        "window_length": window_length,
        "hidden_channels": hidden_channels,
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
        # The comparison table reads n_parameters.
        "n_parameters": sum(p.numel() for p in model.parameters()),
        # Sanity check that the frozen threshold alerts on about 1% of calib.
        "calib_alert_rate": float((calib_scores > threshold).mean()),
        "test_metrics": {
            **threshold_metrics(y_test, test_scores, threshold),
            **ranking_metrics(y_test, test_scores),
        },
        "per_category_recall": recall.reset_index(names="category").to_dict("records"),
        "sanity_check": sanity_check(calib_X, score_calib),
    }
    directory = run_dir("cnn", seed, root=out_root)
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
    parser.add_argument("--hidden-channels", type=int, default=32)
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
            hidden_channels=args.hidden_channels,
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
