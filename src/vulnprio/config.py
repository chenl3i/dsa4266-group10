"""Typed configuration loading.

A run is fully described by its resolved config plus the raw-data snapshot it
was built from. ``load_config`` follows ``extends:`` chains, shallow-merges per
top-level key, and validates the result, so an invalid experiment fails at
startup rather than three hours into a pipeline.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vulnprio.paths import resolve


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Base):
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    external: str = "data/external"
    reports: str = "reports"


class DataConfig(_Base):
    cve_published_min: dt.date
    cve_published_max: dt.date
    drop_rejected: bool = True
    require_cvss_v31: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> DataConfig:
        if self.cve_published_min >= self.cve_published_max:
            raise ValueError("cve_published_min must be before cve_published_max")
        return self


class NVDSourceConfig(_Base):
    """How the NVD corpus is acquired.

    ``feed_year_*`` is deliberately independent of ``data.cve_published_*``:
    feeds are partitioned by CVE *ID* year, so the download range must be able
    to run wider than the study window. See :func:`vulnprio.data.sources.nvd.feed_years`.
    """

    mode: Literal["feed", "api"] = "feed"
    feed_year_min: int = 2015
    feed_year_max: int = 2026

    @model_validator(mode="after")
    def _ordered(self) -> NVDSourceConfig:
        if self.feed_year_min > self.feed_year_max:
            raise ValueError("feed_year_min must be <= feed_year_max")
        return self


class SourcesConfig(_Base):
    nvd: NVDSourceConfig = Field(default_factory=NVDSourceConfig)


class LabelsConfig(_Base):
    observation_cutoff: dt.date
    positive_source: str = "kev"


class SplitConfig(_Base):
    strategy: str = "temporal"
    train_end: dt.date
    valid_end: dt.date
    test_end: dt.date
    embargo_days: int = 0

    @model_validator(mode="after")
    def _ordered(self) -> SplitConfig:
        if not (self.train_end < self.valid_end < self.test_end):
            raise ValueError("split dates must satisfy train_end < valid_end < test_end")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")
        return self


class TextFeatureConfig(_Base):
    vectorizer: str = "tfidf"
    max_features: int = 20_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 5
    apply_leakage_blocklist: bool = True


class FeaturesConfig(_Base):
    blocks: dict[str, bool] = Field(default_factory=dict)
    text: TextFeatureConfig = Field(default_factory=TextFeatureConfig)

    def enabled_blocks(self) -> list[str]:
        """Names of blocks switched on, in a stable (sorted) order."""
        return sorted(name for name, on in self.blocks.items() if on)


class BaselinesConfig(_Base):
    cvss: bool = True
    epss: bool = True
    random: bool = True

    def enabled(self) -> list[str]:
        return [n for n in ("cvss", "epss", "random") if getattr(self, n)]


class ModelsConfig(_Base):
    enabled: list[str] = Field(default_factory=list)


class EvaluationConfig(_Base):
    k_percents: list[float] = Field(default_factory=lambda: [1, 5, 10, 20])
    primary_metric: str = "precision_at_k"
    primary_k_percent: float = 10
    bootstrap_iterations: int = 1000
    bootstrap_confidence: float = 0.95

    @model_validator(mode="after")
    def _valid_k(self) -> EvaluationConfig:
        if any(not 0 < k <= 100 for k in self.k_percents):
            raise ValueError("k_percents must all lie in (0, 100]")
        if self.primary_k_percent not in self.k_percents:
            raise ValueError("primary_k_percent must appear in k_percents")
        return self


class Config(_Base):
    seed: int = 4266
    experiment_name: str = "default"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    data: DataConfig
    labels: LabelsConfig
    split: SplitConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _label_horizon(self) -> Config:
        # Labels must be collected *after* the test window closes, otherwise
        # recent CVEs are under-labelled purely because we stopped looking.
        if self.labels.observation_cutoff < self.split.test_end:
            raise ValueError(
                "labels.observation_cutoff must be >= split.test_end; collecting labels "
                "only up to the split boundary systematically under-labels recent CVEs."
            )
        return self

    def results_dir(self) -> Path:
        return resolve(self.paths.reports) / "results" / self.experiment_name


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge per top-level key.

    Deliberately shallow: a partially-overridden nested block is a common source
    of silent misconfiguration, so an experiment must restate a whole block.
    """
    merged = dict(base)
    merged.update(override)
    return merged


def load_config(path: str | Path) -> Config:
    """Load a YAML config, following ``extends:`` chains, and validate it."""
    path = resolve(path)
    raw = _read_yaml(path)
    parent = raw.pop("extends", None)
    if parent is not None:
        parent_path = (path.parent / parent).resolve()
        base = _read_yaml(parent_path)
        base.pop("extends", None)
        raw = _merge(base, raw)
    return Config.model_validate(raw)


def dump_config(config: Config, path: str | Path) -> Path:
    """Write the fully resolved config next to a run's results."""
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.model_dump(mode="json"), fh, sort_keys=False)
    return out
