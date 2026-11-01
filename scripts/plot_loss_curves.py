"""Plot every saved training loss curve into reports/figures/.

Reads each ``history.csv`` (``epoch, train_loss, tune_loss``) under
``reports/results/`` and draws one figure per model folder, one line per seed, so
the report's curves come from the saved runs rather than from model code. Rerun
after new training runs; existing figures are overwritten.

Usage:
    python scripts/plot_loss_curves.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; no display needed
import matplotlib.pyplot as plt
import pandas as pd

from ugr16ad.paths import FIGURES_DIR, RESULTS_DIR


def figure_name(group: Path) -> str:
    """``lstm`` -> ``lstm``; ``<experiment>/candidates/mlp16`` -> ``mlp_latent16``.

    Every ``experiments.py tune`` folder holds the same candidate names, so runs that
    stopped early (per their ``manifest.json``) get a suffix instead of a clash.
    """
    if group.parent.name != "candidates":
        return group.as_posix().replace("/", "_")
    manifest = json.loads((RESULTS_DIR / group.parent.parent / "manifest.json").read_text())
    training = manifest["training"]
    suffix = "_earlystop" if training["patience"] < training["epochs"] else ""
    return f"mlp_latent{group.name.removeprefix('mlp')}{suffix}"


def main() -> None:
    groups = defaultdict(list)
    for path in sorted(RESULTS_DIR.rglob("seed*/history.csv")):
        groups[figure_name(path.parent.parent.relative_to(RESULTS_DIR))].append(path)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, paths in groups.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, path in enumerate(paths):
            history = pd.read_csv(path)
            colour = f"C{i}"
            seed = path.parent.name
            ax.plot(history.epoch, history.train_loss, color=colour, label=f"{seed} train")
            ax.plot(history.epoch, history.tune_loss, "--", color=colour, label=f"{seed} tune")
            best = history.loc[history.tune_loss.idxmin()]
            ax.plot(best.epoch, best.tune_loss, "o", color=colour)  # weights that get scored
        ax.set(title=f"{name} loss (dot = best tune epoch)", xlabel="Epoch", ylabel="MSE loss")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        out = FIGURES_DIR / f"loss_{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"{out.relative_to(FIGURES_DIR.parent.parent)}  ({len(paths)} seeds)")


if __name__ == "__main__":
    main()
