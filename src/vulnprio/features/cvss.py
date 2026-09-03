"""CVSS v3.1 base-metric features.

This block is the worked example for the other five: it shows the full
FeatureBlock shape - ``required_columns`` -> ``fit`` (training rows only) ->
``transform`` -> ``feature_names``. Copy this file's structure when writing a
new block.

Design note: the eight base metrics are one-hot encoded *in addition to* the
composite base score, not instead of it. The base score is a fixed formula over
those metrics, so a linear model given only the score cannot learn that, say,
network-accessible flaws matter far more for exploitation than availability
impact - even though the formula weights them comparably. Letting the model
re-weight the components is a large part of where any improvement over the CVSS
baseline comes from.
"""

from __future__ import annotations

import pandas as pd

from vulnprio.data import schema
from vulnprio.features.base import FeatureBlock, register_block


@register_block
class CVSSBlock(FeatureBlock):
    name = "cvss"

    def __init__(self, one_hot: bool = True, include_score: bool = True) -> None:
        self.one_hot = one_hot
        self.include_score = include_score
        self._feature_names: list[str] = []

    def required_columns(self) -> list[str]:
        cols = list(schema.CVSS_CATEGORICALS)
        if self.include_score:
            cols.append(schema.CVSS_BASE_SCORE)
        return cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CVSSBlock:
        schema.validate(X, dict.fromkeys(self.required_columns(), "object"), name="cvss input")
        # Category levels come from the CVSS 3.1 specification, not from the
        # observed data, so train and test always produce identical columns even
        # if a rare level happens to be absent from one split.
        self._feature_names = []
        if self.include_score:
            self._feature_names.append(schema.CVSS_BASE_SCORE)
        if self.one_hot:
            for col, levels in schema.CVSS_CATEGORICALS.items():
                self._feature_names.extend(f"{col}__{lvl}" for lvl in levels)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        if self.include_score:
            out[schema.CVSS_BASE_SCORE] = pd.to_numeric(
                X[schema.CVSS_BASE_SCORE], errors="coerce"
            ).astype("float32")
        if self.one_hot:
            for col, levels in schema.CVSS_CATEGORICALS.items():
                values = X[col].astype("string").str.upper()
                for lvl in levels:
                    out[f"{col}__{lvl}"] = (values == lvl).astype("int8")
        return out[self._feature_names]

    def feature_names(self) -> list[str]:
        return list(self._feature_names)
