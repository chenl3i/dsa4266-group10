"""The feature-block contract and registry.

A *block* is a self-contained group of related features (all the CVSS metrics,
all the CWE features, ...) that can be switched on or off from config. Two
reasons this shape is worth the small extra ceremony:

1. **Ablation is the analysis.** "Which signals actually matter?" is answered by
   re-running with one block off. That should be a config edit, not a code edit.
2. **Parallel work.** Blocks share only the schema contract, so group members
   can write ``cwe.py`` and ``text.py`` at the same time without conflicts.

To add a block: create ``features/<name>.py``, subclass :class:`FeatureBlock`,
decorate with ``@register_block``, import it in ``features/__init__.py``, and add
its name under ``features.blocks`` in the config.
"""

from __future__ import annotations

import abc

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_REGISTRY: dict[str, type[FeatureBlock]] = {}


class FeatureBlock(BaseEstimator, TransformerMixin, abc.ABC):
    """A named group of features derived from the analysis table.

    Implemented as a scikit-learn transformer so blocks compose into a Pipeline
    and, critically, so anything *fitted* (vocabulary, category levels, target
    statistics) is fitted on training rows only. Fitting a vectoriser on the full
    corpus before splitting is a leak, and the Pipeline shape makes that mistake
    hard to commit by accident.
    """

    #: Registry key; also the config key under ``features.blocks``.
    name: str

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureBlock:
        """Learn any state from the training rows only."""

    @abc.abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return this block's features, one row per input row, index preserved."""

    @abc.abstractmethod
    def feature_names(self) -> list[str]:
        """Output column names, for coefficient and importance reporting."""

    def required_columns(self) -> list[str]:
        """Input columns this block reads. Used to validate the schema up front."""
        return []


def register_block(cls: type[FeatureBlock]) -> type[FeatureBlock]:
    """Class decorator adding a block to the registry."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`.")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Feature block {name!r} is already registered.")
    _REGISTRY[name] = cls
    return cls


def get_block(name: str) -> type[FeatureBlock]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown feature block {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_blocks() -> list[str]:
    return sorted(_REGISTRY)


def build_blocks(names: list[str], **kwargs) -> list[FeatureBlock]:
    """Instantiate the named blocks, passing per-block kwargs where accepted."""
    blocks: list[FeatureBlock] = []
    for name in names:
        cls = get_block(name)
        block_kwargs = kwargs.get(name, {})
        blocks.append(cls(**block_kwargs))
    return blocks
