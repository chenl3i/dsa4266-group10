"""Description-text features.

Descriptions carry signal the CVSS vector cannot: which software component is
involved, the exploitation mechanism, and the phrasing analysts use for remote
code execution versus a local denial of service.

They also carry this project's worst leak - see :mod:`vulnprio.leakage` item 3.
``apply_leakage_blocklist`` defaults to True, and the block calls
``leakage.scrub_series`` on the description column before vectorising.

Planned approach (plan Task 11):
  - TF-IDF over 1-2 grams, ``min_df=5``, capped at ``max_features``.
  - Vectoriser fitted on the **training split only**. This is the whole reason
    feature blocks are sklearn transformers rather than plain functions.
  - Report the top-weighted terms. They are the most readable evidence of what
    the model learned, and the fastest way to spot a leak the blocklist missed.

A sentence-embedding model over descriptions is a natural extension, but is
deliberately out of scope for the first pass - establish that TF-IDF beats CVSS
before reaching for anything heavier.
"""

from __future__ import annotations

import pandas as pd

from vulnprio.data import schema
from vulnprio.features.base import FeatureBlock, register_block


@register_block
class TextBlock(FeatureBlock):
    name = "text"

    def __init__(
        self,
        max_features: int = 20_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 5,
        apply_leakage_blocklist: bool = True,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.apply_leakage_blocklist = apply_leakage_blocklist

    def required_columns(self) -> list[str]:
        return [schema.DESCRIPTION]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TextBlock:
        raise NotImplementedError("Implemented in plan Task 11 (text features).")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Implemented in plan Task 11 (text features).")

    def feature_names(self) -> list[str]:
        raise NotImplementedError("Implemented in plan Task 11 (text features).")
