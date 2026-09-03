"""Assemble the enabled feature blocks into a single matrix.

Implemented in plan Task 12.
"""

from __future__ import annotations

import pandas as pd

from vulnprio.config import FeaturesConfig
from vulnprio.features.base import FeatureBlock, get_block


def build_feature_blocks(config: FeaturesConfig) -> list[FeatureBlock]:
    """Instantiate the blocks enabled in config, passing per-block settings."""
    blocks: list[FeatureBlock] = []
    for name in config.enabled_blocks():
        cls = get_block(name)
        if name == "text":
            blocks.append(
                cls(
                    max_features=config.text.max_features,
                    ngram_range=tuple(config.text.ngram_range),
                    min_df=config.text.min_df,
                    apply_leakage_blocklist=config.text.apply_leakage_blocklist,
                )
            )
        else:
            blocks.append(cls())
    return blocks


def fit_transform(blocks: list[FeatureBlock], train: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Fit every block on the training rows and return the concatenated matrix."""
    raise NotImplementedError("Implemented in plan Task 12 (feature assembly).")


def transform(blocks: list[FeatureBlock], frame: pd.DataFrame) -> pd.DataFrame:
    """Apply already-fitted blocks to validation or test rows."""
    raise NotImplementedError("Implemented in plan Task 12 (feature assembly).")
