"""A registry for baselines, kept separate from the model registry.

Baselines implement the same :class:`~vulnprio.models.base.Ranker` interface as
models - that is the point, since it forces the evaluation code to treat them
identically. But they must not be *registered* as models: nothing here is learned
from the training data, and letting ``models.enabled: [cvss]`` silently resolve to
the CVSS baseline would let a reference line be reported as a result.

Two registries, one interface.
"""

from __future__ import annotations

from vulnprio.models.base import Ranker

_REGISTRY: dict[str, type[Ranker]] = {}


def register_baseline(cls: type[Ranker]) -> type[Ranker]:
    """Class decorator adding a baseline to the baseline registry."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`.")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Baseline {name!r} is already registered.")
    _REGISTRY[name] = cls
    return cls


def get_baseline(name: str) -> type[Ranker]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown baseline {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_baselines() -> list[str]:
    return sorted(_REGISTRY)
