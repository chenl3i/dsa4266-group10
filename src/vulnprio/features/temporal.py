"""Age and calendar features.

Computed relative to a *reference date* rather than "today", because a feature
whose value changes every time you rerun the pipeline is not reproducible - and
"age as of now" quietly leaks the future into training rows.

Planned features (plan Task 7):
  - ``age_days``: reference_date minus published_date
  - ``publication_year``, ``publication_month``, ``publication_quarter``
  - ``days_since_last_modified``: proxy for how much post-publication analyst
    attention a CVE has attracted. Flagged as a partial leak risk in
    :mod:`vulnprio.leakage` - report the ablation.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from vulnprio.data import schema
from vulnprio.features.base import FeatureBlock, register_block


@register_block
class TemporalBlock(FeatureBlock):
    name = "temporal"

    def __init__(self, reference_date: dt.date | None = None) -> None:
        self.reference_date = reference_date

    def required_columns(self) -> list[str]:
        return [schema.PUBLISHED, schema.LAST_MODIFIED]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TemporalBlock:
        raise NotImplementedError("Implemented in plan Task 7 (temporal features).")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Implemented in plan Task 7 (temporal features).")

    def feature_names(self) -> list[str]:
        raise NotImplementedError("Implemented in plan Task 7 (temporal features).")
