"""Assemble the comparison table from reports/results/<model>/seed<seed>/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from ugr16ad.data import make_part, read_split
from ugr16ad.evaluate import per_category_recall, ranking_metrics, threshold_metrics
from ugr16ad.paths import RESULTS_DIR
from ugr16ad.results import load_run

METRICS = [
    "recall",
    "precision",
    "f1",
    "fpr",
    "alert_rate",
    "average_precision",
    "recall_at_max_fpr",
    "partial_auc",
    "roc_auc",
    "train_seconds",
    "inference_seconds",
]


def load_runs(results_dir: Path = RESULTS_DIR) -> dict:
    """``{(model, seed): (test scores indexed by minute, meta)}`` for every saved run."""
    runs = {}
    for seed_dir in sorted(results_dir.glob("*/seed*")):
        if not (seed_dir / "meta.json").exists():
            continue
        run = load_run(seed_dir)
        test = run.scores[run.scores["part"] == "test"].set_index("timestamp")["score"]
        runs[(run.meta["model"], run.meta["seed"])] = (test, run.meta)
    if not runs:
        raise SystemExit(f"No runs found under {results_dir}")
    return runs


def common_minutes(runs: dict) -> pd.DatetimeIndex:
    """The test minutes every run scored. Stops if the runs disagree.

    Every model must be scored on the same eligible minutes. A mismatch means a
    run used a different window length or part boundary, so stop instead of
    silently shrinking the evaluation set.
    """
    indices = [test.index for test, _ in runs.values()]
    common = indices[0]
    mismatched = [key for key, index in zip(runs, indices, strict=True) if not index.equals(common)]
    if mismatched:
        raise SystemExit(f"Runs scored on different test minutes: {mismatched}")
    return common


def build_tables(
    runs: dict, labels: pd.DataFrame, y_true
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(per_seed, summary, per_category)``.

    ``labels`` is indexed by the common test minutes. Each run is judged at its own
    frozen threshold from ``meta["threshold"]``, which is never refit here.
    """
    rows, category_rows = [], []
    for (model, seed), (test, meta) in runs.items():
        scores = test.loc[labels.index].to_numpy()
        threshold = meta["threshold"]
        row = {"model": model, "seed": seed}
        row |= threshold_metrics(y_true, scores, threshold)
        row |= ranking_metrics(y_true, scores)
        # Full ROC-AUC: the usual summary, next to the stricter partial AUC over FPR 0-1%.
        row["roc_auc"] = float(roc_auc_score(y_true, scores))
        row |= {k: meta.get(k) for k in ("train_seconds", "inference_seconds", "n_parameters")}
        rows.append(row)
        categories = per_category_recall(labels, scores > threshold).reset_index(names="category")
        category_rows.append(categories.assign(model=model, seed=seed))

    per_seed = pd.DataFrame(rows)
    per_seed[METRICS] = per_seed[METRICS].apply(pd.to_numeric, errors="coerce")
    summary = per_seed.groupby("model")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_parameters"] = per_seed.groupby("model")["n_parameters"].first()
    per_category = (
        pd.concat(category_rows).groupby(["model", "category"])[["n_positive", "recall"]].mean()
    )
    return per_seed, summary, per_category


def main(results_dir: Path = RESULTS_DIR) -> None:
    runs = load_runs(results_dir)
    common = common_minutes(runs)
    print(f"{len(runs)} runs, all scored on the same {len(common)} test minutes")

    X, Y = read_split("test")
    test_part = make_part(X, Y)
    labels = test_part.labels.loc[common]
    y_true = test_part.y.loc[common].to_numpy()  # one definition of "attack", in make_part
    print(f"{int(y_true.sum())} attack minutes among them")

    per_seed, summary, per_category = build_tables(runs, labels, y_true)
    per_seed.to_csv(results_dir / "comparison_per_seed.csv", index=False)
    summary.to_csv(results_dir / "comparison_summary.csv")
    per_category.to_csv(results_dir / "comparison_per_category.csv")
    print(summary.round(3).T.to_string())
    print()
    print(per_category.round(3).to_string())


if __name__ == "__main__":
    main()