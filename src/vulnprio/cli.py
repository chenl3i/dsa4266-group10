"""Command-line entry points.

One command per pipeline stage, each reading the same config, so any stage can
be re-run without repeating the ones before it. This matters because ``fetch``
takes tens of minutes against the NVD API while ``evaluate`` takes seconds - and
you will run ``evaluate`` a hundred times.

    vulnprio fetch          --config configs/default.yaml
    vulnprio build-dataset  --config configs/default.yaml
    vulnprio build-features --config configs/default.yaml
    vulnprio evaluate       --config configs/experiments/baseline_cvss.yaml
    vulnprio info

Stage bodies land in plan Tasks 2-17; the command surface and config wiring are
in place now so every task has an obvious place to plug into.
"""

from __future__ import annotations

import typer

from vulnprio.config import load_config
from vulnprio.data.sources.nvd import NVDSource
from vulnprio.logging_utils import get_logger, setup_logging
from vulnprio.paths import resolve

app = typer.Typer(add_completion=False, help="Risk-based vulnerability prioritisation.")
log = get_logger(__name__)

ConfigOption = typer.Option("configs/default.yaml", "--config", "-c", help="Path to a YAML config.")
YearOption = typer.Option(
    None, "--year", help="Restrict the NVD feed pull to these years (repeatable)."
)


@app.command()
def fetch(
    config: str = ConfigOption,
    force: bool = False,
    year: list[int] = YearOption,
) -> None:
    """Download raw NVD, KEV and EPSS snapshots into data/raw/."""
    setup_logging()
    cfg = load_config(config)
    log.info("Fetching sources for experiment %r", cfg.experiment_name)

    nvd_cfg = cfg.sources.nvd
    # --year narrows the pull to specific feed files, so the live smoke test can
    # cost 3 MB instead of 172 MB.
    year_min = min(year) if year else nvd_cfg.feed_year_min
    year_max = max(year) if year else nvd_cfg.feed_year_max

    source = NVDSource(
        resolve(cfg.paths.raw),
        published_min=cfg.data.cve_published_min,
        published_max=cfg.data.cve_published_max,
        mode=nvd_cfg.mode,
        feed_year_min=year_min,
        feed_year_max=year_max,
    )
    snapshot = source.fetch(force=force)
    log.info("NVD snapshot: %d file(s) in %s", len(snapshot.files), snapshot.directory)

    # Not raising: NVD alone is a usable artifact, and blocking on the unwritten
    # sources would make the one stage that does work unreachable.
    log.warning(
        "KEV and EPSS acquisition are not implemented yet (plan Tasks 3-4). "
        "The dataset has features but no labels until KEV lands."
    )


@app.command("build-dataset")
def build_dataset(config: str = ConfigOption) -> None:
    """Parse raw snapshots, attach labels, assign temporal splits."""
    setup_logging()
    cfg = load_config(config)
    log.info("Building dataset: %s to %s", cfg.data.cve_published_min, cfg.data.cve_published_max)
    raise NotImplementedError("Implemented in plan Tasks 5-6 (parsing, labelling, splits).")


@app.command("build-features")
def build_features(config: str = ConfigOption) -> None:
    """Fit enabled feature blocks on train and transform every split."""
    setup_logging()
    cfg = load_config(config)
    log.info("Enabled feature blocks: %s", cfg.features.enabled_blocks())
    raise NotImplementedError("Implemented in plan Task 12 (feature assembly).")


@app.command()
def evaluate(config: str = ConfigOption) -> None:
    """Score every enabled baseline and model, and write the comparison table."""
    setup_logging()
    cfg = load_config(config)
    log.info("Baselines: %s | Models: %s", cfg.baselines.enabled(), cfg.models.enabled)
    raise NotImplementedError("Implemented in plan Tasks 15-17 (evaluation and reporting).")


@app.command()
def info(config: str = ConfigOption) -> None:
    """Print the resolved config and everything currently registered."""
    setup_logging()
    from vulnprio.baselines import available_baselines
    from vulnprio.features.base import available_blocks
    from vulnprio.models.base import available_models

    cfg = load_config(config)
    typer.echo(f"experiment      : {cfg.experiment_name}")
    typer.echo(f"seed            : {cfg.seed}")
    typer.echo(f"cve window      : {cfg.data.cve_published_min} .. {cfg.data.cve_published_max}")
    typer.echo(
        f"split           : train<={cfg.split.train_end} valid<={cfg.split.valid_end} "
        f"test<={cfg.split.test_end} (embargo {cfg.split.embargo_days}d)"
    )
    typer.echo(f"label cutoff    : {cfg.labels.observation_cutoff}")
    typer.echo(f"blocks enabled  : {cfg.features.enabled_blocks()}")
    typer.echo(f"blocks available: {available_blocks()}")
    typer.echo(f"models enabled  : {cfg.models.enabled or '(none)'}")
    typer.echo(f"models available: {available_models() or '(none registered yet)'}")
    typer.echo(f"baselines enabled  : {cfg.baselines.enabled()}")
    typer.echo(f"baselines available: {available_baselines()}")
    typer.echo(f"k percents      : {cfg.evaluation.k_percents}")


if __name__ == "__main__":
    app()
