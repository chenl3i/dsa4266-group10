"""The two registries must stay separate.

Baselines share the Ranker interface with models on purpose - that is what keeps
the evaluation code from treating them differently. But a baseline must never be
selectable via ``models.enabled``, or a reference line could be reported as a
result.
"""

from __future__ import annotations

import pytest

import vulnprio.baselines  # noqa: F401  - imported for registration side effect
from vulnprio.baselines import available_baselines, get_baseline
from vulnprio.models.base import Ranker, available_models, get_model


def test_all_three_baselines_are_registered():
    assert available_baselines() == ["cvss", "epss", "random"]


def test_model_registry_is_empty_until_models_are_added():
    # Importing baselines must not populate the model registry.
    assert available_models() == []


def test_baselines_are_not_reachable_as_models():
    for name in available_baselines():
        with pytest.raises(KeyError):
            get_model(name)


def test_baselines_implement_the_ranker_interface():
    for name in available_baselines():
        assert issubclass(get_baseline(name), Ranker)


def test_unknown_baseline_raises():
    with pytest.raises(KeyError, match="Unknown baseline"):
        get_baseline("nonexistent")
