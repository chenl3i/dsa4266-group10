"""The model contract.

Every model in this project is a **ranker**, not a classifier. The deliverable
is an ordering of CVEs under a patch budget, so no model ever needs to pick a
decision threshold - it needs to produce a score whose ordering is good.

Keeping that distinction in the interface avoids the most common framing error
on imbalanced problems: reporting accuracy or a 0.5-threshold F1 for a task
where the threshold is set by how many people you have to do the patching.

The registry is intentionally empty. To add a model:

  1. Create ``src/vulnprio/models/<name>.py``.
  2. Subclass :class:`Ranker`, set ``name``, decorate with ``@register_model``.
  3. Import it in ``src/vulnprio/models/__init__.py``.
  4. Add ``<name>`` to ``models.enabled`` in your config.

Nothing in the pipeline, CLI, or evaluation code changes.
"""

from __future__ import annotations

import abc

import numpy as np
import pandas as pd

_REGISTRY: dict[str, type[Ranker]] = {}


class Ranker(abc.ABC):
    """Produces a score per CVE; higher means patch sooner."""

    #: Registry key; also the name used in ``models.enabled`` and in reports.
    name: str

    #: Whether ``score`` returns a calibrated probability. Baselines that return
    #: a raw severity score set this to False, which suppresses calibration
    #: plots that would be meaningless for them.
    produces_probabilities: bool = False

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Ranker:
        """Fit on training rows. Unsupervised baselines may ignore ``y``."""

    @abc.abstractmethod
    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return one score per row, higher = higher exploitation risk."""

    def rank(self, X: pd.DataFrame) -> np.ndarray:
        """Row indices ordered from highest to lowest score.

        Ties are broken by the caller's policy in
        :func:`vulnprio.evaluation.metrics.order_by_score`, not here - tie
        handling is an evaluation decision, and baking a policy into each model
        would make models incomparable.
        """
        return np.argsort(-self.score(X), kind="stable")

    def describe(self) -> dict[str, object]:
        """Hyperparameters and fitted state worth recording in the run report."""
        return {"name": self.name}


def register_model(cls: type[Ranker]) -> type[Ranker]:
    """Class decorator adding a ranker to the registry."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`.")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Model {name!r} is already registered.")
    _REGISTRY[name] = cls
    return cls


def get_model(name: str) -> type[Ranker]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Registered: {sorted(_REGISTRY) or '(none yet)'}. "
            "See src/vulnprio/models/README.md for how to add one."
        )
    return _REGISTRY[name]


def available_models() -> list[str]:
    return sorted(_REGISTRY)
