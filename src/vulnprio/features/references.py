"""Reference-URL and reference-tag features.

Read :mod:`vulnprio.leakage` item 4 before enabling this block. References
accumulate *after* publication. A link to an exploit database or a Metasploit
module may be a consequence of exploitation rather than a predictor of it.

The block exists anyway, because excluding these features silently is worse than
including them with an honest ablation. The report must present results both
with and without, and state which it treats as the headline number.

Planned features (plan Task 10):
  - Counts per NVD reference tag (Exploit, Patch, Vendor Advisory, ...).
  - Domain indicators for known exploit-publication sites.
  - ``n_references`` total: a crude proxy for how much attention a CVE drew.
"""

from __future__ import annotations

import pandas as pd

from vulnprio.data import schema
from vulnprio.features.base import FeatureBlock, register_block

NVD_REFERENCE_TAGS: tuple[str, ...] = (
    "Exploit",
    "Patch",
    "Vendor Advisory",
    "Third Party Advisory",
    "Mitigation",
    "VDB Entry",
    "Issue Tracking",
    "Technical Description",
)

EXPLOIT_DOMAINS: tuple[str, ...] = (
    "exploit-db.com",
    "packetstormsecurity.com",
    "rapid7.com",
    "github.com",
    "seclists.org",
    "zerodayinitiative.com",
)


@register_block
class ReferencesBlock(FeatureBlock):
    name = "references"

    def __init__(self, include_exploit_tag: bool = True) -> None:
        # This toggle exists specifically so the leakage ablation is a one-line
        # config change rather than a code edit.
        self.include_exploit_tag = include_exploit_tag

    def required_columns(self) -> list[str]:
        return [schema.REFERENCE_URLS, schema.REFERENCE_TAGS]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ReferencesBlock:
        raise NotImplementedError("Implemented in plan Task 10 (reference features).")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Implemented in plan Task 10 (reference features).")

    def feature_names(self) -> list[str]:
        raise NotImplementedError("Implemented in plan Task 10 (reference features).")
