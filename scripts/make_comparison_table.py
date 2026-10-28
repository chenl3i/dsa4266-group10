"""Assemble the comparison table from reports/results/<model>/seed<seed>/."""

from __future__ import annotations

import pandas as pd

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
    "train_seconds",
    "inference_seconds",
]


def main() -> None:
    runs = {}
    for seed_dir in sorted(RESULTS_DIR.glob("*/seed*")):
        if not (seed_dir / "meta.json").exists():
            continue
        run = load_run(seed_dir)
        test = run.scores[run.scores["part"] == "test"].set_index("timestamp")["score"]
        runs[(run.meta["model"], run.meta["seed"])] = (test, run.meta)
    if not runs:
        raise SystemExit(f"No runs found under {RESULTS_DIR}")

    # Windowed models score fewer minutes, so compare on the minutes every model scored.
    common = None
    for test, _ in runs.values():
        common = test.index if common is None else common.intersection(test.index)
    print(f"{len(runs)} runs, {len(common)} common test minutes")

    X, Y = read_split("test")
    labels = make_part(X, Y).labels.loc[common]
    y_true = (labels > 0).any(axis=1).to_numpy()
    print(f"{int(y_true.sum())} attack minutes among them")

    rows, category_rows = [], []
    for (model, seed), (test, meta) in runs.items():
        scores = test.loc[common].to_numpy()
        threshold = meta["threshold"]  # frozen on calibration; never refit here
        row = {"model": model, "seed": seed}
        row |= threshold_metrics(y_true, scores, threshold)
        row |= ranking_metrics(y_true, scores)
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

    per_seed.to_csv(RESULTS_DIR / "comparison_per_seed.csv", index=False)
    summary.to_csv(RESULTS_DIR / "comparison_summary.csv")
    per_category.to_csv(RESULTS_DIR / "comparison_per_category.csv")
    print(summary.round(3).T.to_string())
    print()
    print(per_category.round(3).to_string())


if __name__ == "__main__":
    main()
