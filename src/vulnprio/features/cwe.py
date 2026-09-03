"""CWE weakness-type features.

The brief asks "are certain types of vulnerabilities disproportionately
targeted?" - this block is how that question becomes a feature.

Raw CWE IDs are too sparse to one-hot directly (900+ IDs with a long tail).
Planned approach (plan Task 8):
  - Map each CVE to its CWE IDs, then roll up to a coarse class via the CWE-1003
    "Weaknesses for Simplified Mapping" view (~40 classes). The lookup table
    lives in ``data/external/cwe_categories.csv`` - committed to git, since it
    is small and hand-checked.
  - One-hot the top-N classes by *training-set* frequency; everything else maps
    to ``other``.
  - Treat the ``NVD-CWE-noinfo`` and ``NVD-CWE-Other`` sentinels as their own
    indicator rather than as missing data - "unclassified by NVD" is itself
    informative, and often correlates with how much analyst attention a CVE got.
"""

from __future__ import annotations

import pandas as pd

from vulnprio.data import schema
from vulnprio.features.base import FeatureBlock, register_block

NVD_CWE_SENTINELS: tuple[str, ...] = ("NVD-CWE-noinfo", "NVD-CWE-Other")


@register_block
class CWEBlock(FeatureBlock):
    name = "cwe"

    def __init__(self, top_n: int = 40, rollup: bool = True) -> None:
        self.top_n = top_n
        self.rollup = rollup

    def required_columns(self) -> list[str]:
        return [schema.CWE_IDS]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CWEBlock:
        raise NotImplementedError("Implemented in plan Task 8 (CWE features).")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Implemented in plan Task 8 (CWE features).")

    def feature_names(self) -> list[str]:
        raise NotImplementedError("Implemented in plan Task 8 (CWE features).")
