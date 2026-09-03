"""Config validation tests.

These exist because a misconfigured split is silent: the pipeline runs, produces
numbers, and the numbers are wrong. Every guard below corresponds to a mistake
that would otherwise only surface as a suspiciously good result.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from vulnprio.config import load_config

BASE = {
    "seed": 4266,
    "data": {
        "cve_published_min": "2016-01-01",
        "cve_published_max": "2024-12-31",
    },
    "labels": {"observation_cutoff": "2025-06-30"},
    "split": {
        "train_end": "2022-12-31",
        "valid_end": "2023-06-30",
        "test_end": "2023-12-31",
        "embargo_days": 180,
    },
    "evaluation": {"k_percents": [1, 5, 10], "primary_k_percent": 10},
}


def write(tmp_path, payload, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_loads_the_shipped_default():
    cfg = load_config("configs/default.yaml")
    assert cfg.seed == 4266
    assert "cvss" in cfg.features.enabled_blocks()


def test_experiment_extends_and_overrides_default():
    cfg = load_config("configs/experiments/baseline_cvss.yaml")
    assert cfg.experiment_name == "baseline_cvss_only"
    # The features block is restated wholesale, so only cvss survives.
    assert cfg.features.enabled_blocks() == ["cvss"]
    # Unstated keys still come from the parent.
    assert cfg.seed == 4266


def test_rejects_label_cutoff_before_test_end(tmp_path):
    payload = {**BASE, "labels": {"observation_cutoff": "2023-01-01"}}
    with pytest.raises(ValidationError, match="observation_cutoff"):
        load_config(write(tmp_path, payload))


def test_rejects_out_of_order_split_dates(tmp_path):
    payload = {**BASE, "split": {**BASE["split"], "valid_end": "2021-01-01"}}
    with pytest.raises(ValidationError, match="train_end < valid_end < test_end"):
        load_config(write(tmp_path, payload))


def test_rejects_primary_k_absent_from_k_percents(tmp_path):
    payload = {**BASE, "evaluation": {"k_percents": [1, 5], "primary_k_percent": 10}}
    with pytest.raises(ValidationError, match="primary_k_percent"):
        load_config(write(tmp_path, payload))


def test_rejects_unknown_keys(tmp_path):
    payload = {**BASE, "nonsense_key": True}
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, payload))


def test_enabled_blocks_is_sorted_and_filtered():
    cfg = load_config("configs/default.yaml")
    blocks = cfg.features.enabled_blocks()
    assert blocks == sorted(blocks)
    assert all(cfg.features.blocks[b] for b in blocks)


def test_sources_defaults_to_feed_mode_when_absent():
    """Existing configs predate the sources block and must still load."""
    cfg = load_config("configs/default.yaml")
    assert cfg.sources.nvd.mode == "feed"


def test_feed_range_may_be_wider_than_the_study_window():
    """The two ranges answer different questions and must not be cross-validated."""
    cfg = load_config("configs/default.yaml")
    assert cfg.sources.nvd.feed_year_min <= cfg.data.cve_published_min.year
    assert cfg.sources.nvd.feed_year_max >= cfg.data.cve_published_max.year


def test_rejects_an_inverted_feed_year_range(tmp_path):
    from vulnprio.config import NVDSourceConfig

    with pytest.raises(ValidationError):
        NVDSourceConfig(feed_year_min=2026, feed_year_max=2015)
