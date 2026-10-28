"""Recorded PCA/MLP tuning runs; final evaluation consumes frozen checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from ugr16ad.data import FIT_ROWS, TUNE_FRACTION, make_part, read_split
from ugr16ad.evaluate import (
    fit_threshold,
    per_category_recall,
    ranking_metrics,
    threshold_metrics,
)
from ugr16ad.models.mlp import MLPAutoencoder
from ugr16ad.models.pca import PCAScorer
from ugr16ad.paths import RESULTS_DIR, UGR16_DIR
from ugr16ad.preprocess import Preprocessor
from ugr16ad.results import run_dir, save_run
from ugr16ad.synthetic import SANITY_GRID, SANITY_RATE, SANITY_SEED, sanity_check
from ugr16ad.train import score_autoencoder, set_seed, train_autoencoder
from ugr16ad.windows import eligible_positions

TRAINING = dict(epochs=50, batch_size=256, lr=1e-3, patience=5, device="cpu")
SEEDS = (0, 1, 2)
HIDDEN = (64, 32)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2))


def elbow_index(dimensions, losses):
    """Largest improvement below the endpoint chord; fall back to minimum loss.

    This rule is fixed before running the grid. It is a practical operational
    definition of the proposal's elbow, not proof of optimal detection.
    """
    x, y = np.asarray(dimensions, dtype=float), np.asarray(losses, dtype=float)
    if len(x) < 3 or y[0] <= y[-1]:
        return int(np.argmin(y))
    benefit = (y[0] - y) / (y[0] - y[-1]) - (x - x[0]) / (x[-1] - x[0])
    return int(np.argmax(benefit)) if benefit.max() > 1e-8 else int(np.argmin(y))


def configure_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def train_mlp(fit, tune, latent, seed):
    set_seed(seed)
    model = MLPAutoencoder(fit.shape[1], HIDDEN, latent)
    return train_autoencoder(model, fit, tune, seed=seed, **TRAINING)


def tune_run(root: Path, tune_fraction: float):
    root.mkdir(parents=True, exist_ok=False)
    configure_cpu()
    X, Y = read_split("train")  # Deliberately do not load test data during selection.
    cut = FIT_ROWS + int((len(X) - FIT_ROWS) * tune_fraction)
    pre = Preprocessor().fit(X.iloc[:FIT_ROWS])
    fit, tune = pre.transform(X.iloc[:FIT_ROWS]), pre.transform(X.iloc[FIT_ROWS:cut])
    if not np.isfinite(fit).all() or not np.isfinite(tune).all():
        raise ValueError("Nonfinite preprocessed values")
    manifest = {
        "fit_rows": FIT_ROWS,
        "tune_fraction": tune_fraction,
        "partition_counts": {"fit": len(fit), "tune": len(tune), "calib": len(X) - cut},
        "feature_names": pre.feature_names_,
        "training": TRAINING,
        "seeds": SEEDS,
        "hidden_dims": HIDDEN,
        "threads": 1,
        "deterministic_algorithms": True,
        "shuffle": "fitting samples only",
        "selection_rule": "maximum normalized distance below endpoint chord; fallback minimum MSE",
        "selection_status": "fixed by the recorded elbow rule before test evaluation",
        "versions": {
            p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "torch")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "source_sha256": {
            str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.rglob("*.py")
        },
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(UGR16_DIR.glob("*train.csv"))
        },
    }
    write_json(root / "manifest.json", manifest)
    joblib.dump(pre, root / "preprocessor.joblib")
    spectrum = PCAScorer(fit.shape[1]).fit(fit).explained_variance_ratio_
    cumulative = np.cumsum(spectrum)
    pd.DataFrame(
        {"components": np.arange(1, len(spectrum) + 1), "cumulative_variance": cumulative}
    ).to_csv(root / "pca_variance.csv", index=False)
    ranks = {
        min(int(np.searchsorted(cumulative, q)) + 1, fit.shape[1] - 1) for q in (0.8, 0.9, 0.95)
    }
    center = min(int(np.searchsorted(cumulative, 0.9)) + 1, fit.shape[1] - 1)
    ranks.update((max(1, center - 4), min(fit.shape[1] - 1, center + 4)))
    latents = [n for n in (4, 8, 16, 32) if n < fit.shape[1]]
    write_json(root / "grid.json", {"pca_ranks": sorted(ranks), "mlp_latents": latents})
    rows, fitted = [], {}
    for rank in sorted(ranks):
        started = time.perf_counter()
        model = PCAScorer(rank).fit(fit)
        seconds = time.perf_counter() - started
        scores, _ = model.score(tune)
        rows.append(
            dict(
                model="pca",
                dimension=rank,
                seed=0,
                tune_mse=float(scores.mean()),
                train_seconds=seconds,
                best_epoch=None,
            )
        )
        fitted[rank] = model
    for latent in latents:
        for seed in SEEDS:
            print(f"MLP latent={latent}, seed={seed}", flush=True)
            result = train_mlp(fit, tune, latent, seed)
            directory = root / "candidates" / f"mlp{latent}" / f"seed{seed}"
            directory.mkdir(parents=True)
            result.history.to_csv(directory / "history.csv", index=False)
            torch.save(result.model.state_dict(), directory / "weights.pt")
            rows.append(
                dict(
                    model="mlp",
                    dimension=latent,
                    seed=seed,
                    tune_mse=float(result.history.tune_loss.min()),
                    train_seconds=result.seconds,
                    best_epoch=result.best_epoch,
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(root / "tuning.csv", index=False)
    selected = {}
    for name in ("pca", "mlp"):
        curve = table[table.model == name].groupby("dimension").tune_mse.mean()
        selected[name] = int(curve.index[elbow_index(curve.index, curve.values)])
    write_json(root / "selected.json", selected)  # Freeze before any calibration/test scoring.
    joblib.dump(fitted[selected["pca"]], root / "pca.joblib")

    # Repeat training, then also verify saved weights reproduce scores exactly.
    dimension = selected["mlp"]
    repeat = train_mlp(fit, tune, dimension, SEEDS[0])
    original_dir = root / "candidates" / f"mlp{dimension}" / "seed0"
    original = MLPAutoencoder(fit.shape[1], HIDDEN, dimension)
    original.load_state_dict(torch.load(original_dir / "weights.pt", weights_only=True))
    expected = pd.read_csv(original_dir / "history.csv")
    history_equal = np.allclose(
        expected.to_numpy(), repeat.history.to_numpy(), rtol=1e-12, atol=1e-12
    )
    weight_equal = all(
        torch.equal(v, original.state_dict()[k]) for k, v in repeat.model.state_dict().items()
    )
    score_equal = np.array_equal(
        score_autoencoder(original, tune)[0], score_autoencoder(repeat.model, tune)[0]
    )
    pca_repeat = PCAScorer(selected["pca"]).fit(fit)
    pca_equal = np.array_equal(pca_repeat.score(tune)[0], fitted[selected["pca"]].score(tune)[0])
    audit = dict(
        history_matches=bool(history_equal),
        weights_identical=weight_equal,
        scores_identical=score_equal,
        pca_repeat_identical=pca_equal,
        scope="Same CPU, software versions and one thread; not a cross-platform guarantee",
    )
    write_json(root / "reproducibility.json", audit)
    if not all((history_equal, weight_equal, score_equal, pca_equal)):
        raise RuntimeError("Reproducibility audit failed; inspect saved outputs")
    print(table.to_string(index=False), flush=True)
    print("Selected:", selected, "\nReproducibility:", audit, flush=True)


def evaluate_run(root, window, *, results_root=None):
    configure_cpu()
    manifest = json.loads((root / "manifest.json").read_text())
    selected = json.loads((root / "selected.json").read_text())
    for name, digest in manifest["data_sha256"].items():
        if hashlib.sha256((UGR16_DIR / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Training data changed since tuning")
    results_root = RESULTS_DIR if results_root is None else Path(results_root)
    targets = [
        run_dir(name, seed, root=results_root)
        for name, seeds in (("pca", (0,)), ("mlp", manifest["seeds"]))
        for seed in seeds
    ]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"Run already exists: {target}; choose a new --results-root")
    destination = root / f"evaluation-window{window}"
    destination.mkdir(exist_ok=False)
    write_json(
        destination / "protocol.json",
        dict(
            window=window,
            rate=SANITY_RATE,
            grid=SANITY_GRID,
            synthetic_seed=SANITY_SEED,
            synthetic_partition="calib",
            synthetic_metric="roc_auc",
            evaluation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            quantile=0.99,
        ),
    )
    X, Y = read_split("train")
    fit_rows = manifest["fit_rows"]
    cut = fit_rows + manifest["partition_counts"]["tune"]
    parts = {
        "tune": make_part(X.iloc[fit_rows:cut], Y.iloc[fit_rows:cut]),
        "calib": make_part(X.iloc[cut:], Y.iloc[cut:]),
        "test": make_part(*read_split("test")),
    }
    pre = joblib.load(root / "preprocessor.joblib")
    positions = {
        key: eligible_positions(part.segment.to_numpy(), window) for key, part in parts.items()
    }
    if any(len(p) == 0 for p in positions.values()):
        raise ValueError("Window leaves an empty scoring partition")
    inputs = {key: pre.transform(part.X)[positions[key]] for key, part in parts.items()}
    calib_X = pre.transform(parts["calib"].X)
    rows = []
    tuning_table = pd.read_csv(root / "tuning.csv")
    for name, seeds in (("pca", (0,)), ("mlp", manifest["seeds"])):
        for seed in seeds:
            directory = run_dir(name, seed, root=results_root)
            if name == "pca":
                model = joblib.load(root / "pca.joblib")
                score = model.score
            else:
                model = MLPAutoencoder(
                    len(pre.feature_names_), tuple(manifest["hidden_dims"]), selected["mlp"]
                )
                source = root / "candidates" / f"mlp{selected['mlp']}" / f"seed{seed}"
                model.load_state_dict(torch.load(source / "weights.pt", weights_only=True))

                def score(x, model=model):
                    return score_autoencoder(model, x)

            arrays, frames, errors, timings = {}, [], [], {}
            for key, values in inputs.items():
                started = time.perf_counter()
                arrays[key], error = score(values)
                timings[key] = time.perf_counter() - started
                frames.append(
                    pd.DataFrame(
                        dict(
                            timestamp=parts[key].X.index[positions[key]],
                            part=key,
                            score=arrays[key],
                        )
                    )
                )
                errors.append(error)
            threshold = fit_threshold(arrays["calib"])
            y = parts["test"].y.iloc[positions["test"]].to_numpy()
            metrics = {
                **threshold_metrics(y, arrays["test"], threshold),
                **ranking_metrics(y, arrays["test"]),
            }
            if len(np.unique(y)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y, arrays["test"]))
            else:
                metrics["roc_auc"] = float("nan")

            def score_calib(values, score=score):
                pos = positions["calib"]
                return score(values[pos])[0], pos

            synthetic = sanity_check(calib_X, score_calib)
            meta = dict(
                model=name,
                seed=seed,
                threshold=threshold,
                dimension=selected[name],
                window=window,
                metrics=metrics,
                inference_seconds=timings,
                sanity_check=synthetic,
                manifest=os.path.relpath((root / "manifest.json").resolve(), directory.resolve()),
                training=manifest["training"] if name == "mlp" else {"svd_solver": "full"},
                train_seconds=float(
                    tuning_table[
                        (tuning_table.model == name)
                        & (tuning_table.dimension == selected[name])
                        & (tuning_table.seed == seed)
                    ].train_seconds.iloc[0]
                ),
                parameter_count=sum(p.numel() for p in model.parameters())
                if name == "mlp"
                else None,
                scored_rows={key: len(value) for key, value in inputs.items()},
                calibration_alert_rate=float((arrays["calib"] > threshold).mean()),
            )
            save_run(
                directory,
                scores=pd.concat(frames, ignore_index=True),
                meta=meta,
                feature_errors=np.concatenate(errors),
            )
            per_category_recall(
                parts["test"].labels.iloc[positions["test"]], arrays["test"] > threshold
            ).to_csv(directory / "categories.csv")
            pd.DataFrame(synthetic).to_csv(directory / "sanity_check.csv", index=False)
            rows.append(
                dict(
                    model=name,
                    seed=seed,
                    threshold=threshold,
                    train_seconds=meta["train_seconds"],
                    inference_seconds=timings["test"],
                    **metrics,
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(destination / "comparison.csv", index=False)
    table.groupby("model").agg(
        {
            key: ["mean", "std"]
            for key in (
                "recall",
                "precision",
                "f1",
                "fpr",
                "alert_rate",
                "average_precision",
                "roc_auc",
                "recall_at_max_fpr",
                "partial_auc",
                "train_seconds",
                "inference_seconds",
            )
        }
    ).to_csv(destination / "summary.csv")
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tune", "evaluate"))
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Experiment folder containing checkpoints and summary tables",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_DIR,
        help="Shared model/seed results root (evaluation only)",
    )
    parser.add_argument("--tune-fraction", type=float, default=TUNE_FRACTION)
    parser.add_argument("--window", type=int)
    args = parser.parse_args()
    if not 0 < args.tune_fraction < 1:
        parser.error("tune-fraction must be between 0 and 1")
    if args.mode == "tune":
        tune_run(args.output, args.tune_fraction)
    else:
        if args.window is None or args.window < 1:
            parser.error("evaluate requires the agreed --window (positive integer)")
        evaluate_run(args.output, args.window, results_root=args.results_root)


if __name__ == "__main__":
    main()
