"""Blind validation sweep for every neural autoencoder, and the ablation table.

Every config is trained on seeds 0-2 with the model's run script in ``--blind`` mode,
so test is never scored here. Configs are ranked by ``selection_score``, fixed before
stage 1, so one rule picks the settings of all five models and the ablation tables
are comparable. Each stage builds on the summary of the earlier ones:

1. width x latent grid at the model's base learning rate;
2. the pick so far with the model's capacity/regularisation variants (LSTM/GRU: 2
   layers, dropout 0 and 0.2; windowed MLP: the other dropout/noise combos; MLP: a
   single-layer and a deeper encoder). The CNN has no stage 2;
3. the top 2 so far with two learning rates around the base.

Run the stages in order 1 -> 2 -> 3: stage 2 builds on the current pick, so re-running it
after stage 3 would expand the new pick instead of the one stage 3 started from.

Runs live under ``reports/runs-sweep/<model>/<tag>/<folder>/seed*`` so models never
overwrite each other and a finished config is skipped on rerun.

Run: ./.venv/Scripts/python.exe scripts/sweep.py --model lstm --stage 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ugr16ad.paths import REPORTS_DIR
from ugr16ad.selection import selection_score

SWEEP_DIR = REPORTS_DIR / "runs-sweep"  # gitignored, and outside RESULTS_DIR
SCRIPTS_DIR = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
LATENTS = (4, 8, 16, 32)


@dataclass(frozen=True)
class Spec:
    script: str
    extra: tuple[str, ...]
    folder: str  # what the run script writes under --out-root
    base: dict
    abbr: dict  # setting -> tag prefix, in tag order
    stage1: tuple[dict, ...]
    stage2: Callable[[dict], list[dict]] | None  # pick -> configs; None: no stage 2
    stage3_lrs: tuple[float, ...]

    @property
    def settings(self) -> tuple[str, ...]:
        return tuple(self.base)


def _grid(base: dict, key: str, widths) -> tuple[dict, ...]:
    return tuple({**base, key: w, "latent_dim": z} for w in widths for z in LATENTS)


def _recurrent_stage2(pick: dict) -> list[dict]:
    return [{**pick, "num_layers": 2, "dropout": d} for d in (0.0, 0.2)]


def _windowed_mlp_stage2(pick: dict) -> list[dict]:
    combos = [(0.0, 0.0), (0.0, 0.1), (0.2, 0.0)]  # {0, 0.2} x {0, 0.1} minus the base
    return [{**pick, "dropout": d, "noise_std": n} for d, n in combos]


def _mlp_stage2(pick: dict) -> list[dict]:
    w = pick["hidden_dims"][0]
    return [{**pick, "hidden_dims": dims} for dims in ((w,), (2 * w, w, w // 2))]


RECURRENT_BASE = {"hidden_dim": 64, "latent_dim": 16, "num_layers": 1, "dropout": 0.0, "lr": 1e-3}
CNN_BASE = {"hidden_channels": 32, "latent_dim": 16, "lr": 1e-3}
WINDOWED_MLP_BASE = {
    "hidden_dim": 64,
    "latent_dim": 16,
    "dropout": 0.2,
    "noise_std": 0.1,
    "lr": 1e-4,
}
MLP_BASE = {"hidden_dims": (64, 32), "latent_dim": 16, "lr": 1e-3}


def _recurrent(cell: str) -> Spec:
    return Spec(
        script="run_lstm.py",
        extra=("--cell", cell),
        folder=cell,
        base=RECURRENT_BASE,
        abbr={"hidden_dim": "h", "latent_dim": "z", "num_layers": "l", "dropout": "d", "lr": "lr"},
        stage1=_grid(RECURRENT_BASE, "hidden_dim", (32, 64, 128)),
        stage2=_recurrent_stage2,
        stage3_lrs=(3e-4, 3e-3),
    )


SPECS = {
    "lstm": _recurrent("lstm"),
    "gru": _recurrent("gru"),
    "cnn": Spec(
        script="run_cnn.py",
        extra=(),
        folder="cnn",
        base=CNN_BASE,
        abbr={"hidden_channels": "c", "latent_dim": "z", "lr": "lr"},
        stage1=_grid(CNN_BASE, "hidden_channels", (16, 32, 64)),
        stage2=None,
        stage3_lrs=(3e-4, 3e-3),
    ),
    "windowed_mlp": Spec(
        script="run_windowed_mlp.py",
        extra=(),
        folder="windowed_mlp",
        base=WINDOWED_MLP_BASE,
        abbr={"hidden_dim": "h", "latent_dim": "z", "dropout": "d", "noise_std": "n", "lr": "lr"},
        stage1=_grid(WINDOWED_MLP_BASE, "hidden_dim", (32, 64, 128)),
        stage2=_windowed_mlp_stage2,
        stage3_lrs=(3e-5, 3e-4),
    ),
    "mlp": Spec(
        script="run_mlp.py",
        extra=(),
        folder="mlp",
        base=MLP_BASE,
        abbr={"hidden_dims": "h", "latent_dim": "z", "lr": "lr"},
        stage1=_grid(MLP_BASE, "hidden_dims", ((32, 16), (64, 32), (128, 64))),
        stage2=_mlp_stage2,
        stage3_lrs=(3e-4, 3e-3),
    ),
}


def _fmt(value) -> str:
    if isinstance(value, (tuple, list)):
        return "-".join(str(v) for v in value)
    return str(value)


def tag(config: dict, spec: Spec) -> str:
    return "_".join(f"{a}{_fmt(config[k])}" for k, a in spec.abbr.items())


def load_runs(model: str, root: Path = SWEEP_DIR) -> pd.DataFrame:
    """One row per finished seed, read from ``meta.json`` only."""
    spec = SPECS[model]
    rows = []
    for path in sorted((root / model).glob(f"*/{spec.folder}/seed*/meta.json")):
        meta = json.loads(path.read_text())
        row = {k: meta[k] for k in (*spec.settings, "seed", "n_parameters", "best_epoch")}
        # JSON has no tuples; groupby and tag need hashable, order-stable values.
        row = {k: tuple(v) if isinstance(v, list) else v for k, v in row.items()}
        row |= {"tag": path.parents[2].name, "best_tune_loss": meta["best_tune_loss"]}
        for c in meta["tune_sanity"]:
            cell_name = f"{c['magnitude']:g}x{c['n_features']}"
            row |= {f"ap_{cell_name}": c["ap"], f"recall_{cell_name}": c["recall"]}
        row["selection_score"] = selection_score(meta)
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame, model: str) -> pd.DataFrame:
    """Mean and std over seeds per config, ranked, with the tie rule applied.

    Configs within one pooled seed-std of the best mean are tied, since that gap
    is within seed noise; the tie goes to the fewest parameters ("keep models
    simple"). Only configs with every seed finished are ranked.
    """
    settings = list(SPECS[model].settings)
    metrics = [c for c in runs.columns if c not in (*settings, "seed", "tag", "n_parameters")]
    grouped = runs.groupby("tag")
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = grouped[[*settings, "n_parameters"]].first().join(summary)
    summary["n_seeds"] = grouped.size()
    summary = summary[summary.n_seeds == len(SEEDS)]
    if summary.empty:
        return summary
    summary = summary.sort_values("selection_score_mean", ascending=False)

    pooled_std = np.sqrt((summary.selection_score_std**2).mean())
    best = summary.selection_score_mean.iloc[0]
    summary["tied"] = summary.selection_score_mean >= best - pooled_std
    summary["pick"] = False
    summary.loc[summary[summary.tied].n_parameters.idxmin(), "pick"] = True
    return summary


def stage_configs(stage: int, summary: pd.DataFrame | None, model: str) -> list[dict]:
    spec = SPECS[model]
    if stage == 1:
        return [dict(c) for c in spec.stage1]
    if stage == 2 and spec.stage2 is None:
        raise ValueError(f"{model} has no stage 2")
    if summary is None or summary.empty:
        raise ValueError(f"stage {stage} needs a finished earlier stage")
    settings = list(spec.settings)
    ranked = summary.sort_values(["pick", "selection_score_mean"], ascending=False)
    if stage == 2:
        return spec.stage2(ranked.iloc[0][settings].to_dict())
    top2 = [row[settings].to_dict() for _, row in ranked.head(2).iterrows()]
    configs = [{**c, "lr": lr} for c in top2 for lr in spec.stage3_lrs]
    # Top 2 differing only in lr expand to the same configs; launching one tag twice
    # concurrently would have two processes writing the same directory.
    return list({tag(c, spec): c for c in configs}.values())


def build_args(config: dict, model: str, threads: int, out_root: Path) -> list[str]:
    spec = SPECS[model]
    args = [sys.executable, "-u", str(SCRIPTS_DIR / spec.script), "--blind", *spec.extra]
    args += ["--seed", *map(str, SEEDS), "--threads", str(threads), "--out-root", str(out_root)]
    for key in spec.settings:
        value = config[key]
        values = value if isinstance(value, (tuple, list)) else [value]
        args += [f"--{key.replace('_', '-')}", *map(str, values)]
    return args


def launch(config: dict, model: str, threads: int, root: Path = SWEEP_DIR) -> None:
    spec = SPECS[model]
    out_root = root / model / tag(config, spec)
    if all((out_root / spec.folder / f"seed{s}" / "meta.json").exists() for s in SEEDS):
        return  # resumable: finished configs are skipped
    out_root.mkdir(parents=True, exist_ok=True)
    args = build_args(config, model, threads, out_root)
    with open(out_root / f"{model}.log", "w") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(SPECS), required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3])
    parser.add_argument("--summary-only", action="store_true", help="rebuild the table, no runs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    model = args.model

    if not args.summary_only:
        if args.stage is None:
            parser.error("--stage is required unless --summary-only")
        runs = load_runs(model)
        try:
            configs = stage_configs(
                args.stage, summarise(runs, model) if len(runs) else None, model
            )
        except ValueError as exc:
            parser.exit(1, f"{model}: {exc}\n")
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(lambda c: launch(c, model, args.threads), configs))

    summary = summarise(load_runs(model), model)
    if summary.empty:
        print(f"{model}: no config has all {len(SEEDS)} seeds yet")
        return
    summary.to_csv(SWEEP_DIR / model / "summary.csv")
    columns = [*SPECS[model].settings, "n_parameters", "selection_score_mean", "tied", "pick"]
    print(summary[columns])


if __name__ == "__main__":
    main()
