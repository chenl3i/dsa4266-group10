"""Raw NVD/KEV/EPSS payloads -> tidy parquet in the interim layer.

Parsing is separated from downloading so the expensive network step runs once
and parser changes can be re-run against a fixed snapshot. Every function here
is pure with respect to the filesystem snapshot it is handed.

Implemented in Task 5 of the plan.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_nvd(snapshot_dir: Path) -> pd.DataFrame:
    """Flatten NVD CVE records into one row per CVE, matching ``schema.NVD_COLUMNS``.

    Notable shape problems the implementation has to handle:
      - CVSS v3.1 may be absent (older or unscored CVEs) -> drop or impute per config.
      - ``cvssMetricV31`` is a list; NVD's own (nvd@nist.gov) entry is authoritative.
      - ``weaknesses`` can hold ``NVD-CWE-noinfo`` / ``NVD-CWE-Other`` sentinels.
      - ``configurations`` nests arbitrarily deep; CPE URIs need parsing to
        vendor/product, not string matching.
    """
    raise NotImplementedError("Implemented in plan Task 5 (NVD parsing).")


def parse_kev(snapshot_dir: Path) -> pd.DataFrame:
    """Return ``[cve_id, kev_date_added, vendor_project, product, ransomware_use]``."""
    raise NotImplementedError("Implemented in plan Task 5 (KEV parsing).")


def parse_epss(snapshot_dir: Path) -> pd.DataFrame:
    """Return ``[cve_id, epss_score, epss_percentile, score_date]``."""
    raise NotImplementedError("Implemented in plan Task 5 (EPSS parsing).")
