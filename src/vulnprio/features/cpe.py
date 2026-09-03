"""Affected-software features, derived from CPE applicability statements.

Intuition worth testing explicitly in the EDA: attackers target *widely
deployed* software. A critical flaw in an obscure library is less attractive
than a moderate flaw in Windows or Chrome. CVSS captures none of this, which
makes it one of the more promising places to beat the baseline.

Planned features (plan Task 9):
  - ``n_affected_products``, ``n_affected_vendors``: breadth of impact.
  - Vendor prevalence, encoded as the vendor's share of *training-set* CVEs.
    Must be fitted on training rows only - computing it over the full corpus is
    target-adjacent leakage.
  - Indicator flags for a small, pre-declared set of high-deployment vendors.
  - ``is_os`` / ``is_application`` / ``is_hardware`` from the CPE part field.

Do not reach for out-of-the-box target encoding on vendor without nested
cross-validation - at a 0.5% positive rate it overfits violently.
"""

from __future__ import annotations

import pandas as pd

from vulnprio.data import schema
from vulnprio.features.base import FeatureBlock, register_block

HIGH_DEPLOYMENT_VENDORS: tuple[str, ...] = (
    "microsoft",
    "google",
    "apple",
    "adobe",
    "oracle",
    "cisco",
    "linux",
    "apache",
    "mozilla",
    "vmware",
)

#: CPE 2.3 part field values.
CPE_PARTS: dict[str, str] = {"o": "is_os", "a": "is_application", "h": "is_hardware"}


@register_block
class CPEBlock(FeatureBlock):
    name = "cpe"

    def __init__(self, top_n_vendors: int = 50) -> None:
        self.top_n_vendors = top_n_vendors

    def required_columns(self) -> list[str]:
        return [schema.CPE_VENDORS, schema.CPE_PRODUCTS]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CPEBlock:
        raise NotImplementedError("Implemented in plan Task 9 (CPE features).")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Implemented in plan Task 9 (CPE features).")

    def feature_names(self) -> list[str]:
        raise NotImplementedError("Implemented in plan Task 9 (CPE features).")
