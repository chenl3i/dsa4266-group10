"""Shared fixtures.

Everything here is synthetic and tiny. Tests must never depend on a downloaded
NVD snapshot: the corpus is 2GB, it changes daily, and a test that needs the
network is a test nobody runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vulnprio.data import schema


@pytest.fixture
def tiny_cves() -> pd.DataFrame:
    """Twelve CVEs spanning 2020-2023, with the full NVD column contract."""
    n = 12
    published = pd.to_datetime(
        [
            "2020-01-15",
            "2020-06-01",
            "2021-03-10",
            "2021-09-20",
            "2022-02-05",
            "2022-07-14",
            "2022-11-30",
            "2023-01-08",
            "2023-04-22",
            "2023-08-11",
            "2023-10-02",
            "2023-12-15",
        ]
    )
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            schema.CVE_ID: pd.Series([f"CVE-2020-{1000 + i}" for i in range(n)], dtype="string"),
            schema.PUBLISHED: published,
            schema.LAST_MODIFIED: published + pd.Timedelta(days=30),
            schema.DESCRIPTION: pd.Series(
                [f"A buffer overflow in component {i} allows remote attackers." for i in range(n)],
                dtype="string",
            ),
            schema.CVSS_BASE_SCORE: rng.choice([4.3, 7.5, 9.8], size=n),
            schema.CVSS_VECTOR: pd.Series(
                ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"] * n, dtype="string"
            ),
            schema.CVSS_SEVERITY: pd.Series(["HIGH"] * n, dtype="string"),
            schema.CVSS_ATTACK_VECTOR: pd.Series(["NETWORK"] * n, dtype="string"),
            schema.CVSS_ATTACK_COMPLEXITY: pd.Series(["LOW"] * n, dtype="string"),
            schema.CVSS_PRIVILEGES_REQUIRED: pd.Series(["NONE"] * n, dtype="string"),
            schema.CVSS_USER_INTERACTION: pd.Series(["NONE"] * n, dtype="string"),
            schema.CVSS_SCOPE: pd.Series(["UNCHANGED"] * n, dtype="string"),
            schema.CVSS_CONF_IMPACT: pd.Series(["HIGH"] * n, dtype="string"),
            schema.CVSS_INTEG_IMPACT: pd.Series(["HIGH"] * n, dtype="string"),
            schema.CVSS_AVAIL_IMPACT: pd.Series(["HIGH"] * n, dtype="string"),
            schema.CWE_IDS: [["CWE-787"] for _ in range(n)],
            schema.CPE_VENDORS: [["microsoft"] for _ in range(n)],
            schema.CPE_PRODUCTS: [["windows"] for _ in range(n)],
            schema.REFERENCE_URLS: [["https://example.com/advisory"] for _ in range(n)],
            schema.REFERENCE_TAGS: [["Vendor Advisory"] for _ in range(n)],
        }
    )


@pytest.fixture
def tiny_kev() -> pd.DataFrame:
    """Three KEV entries. One is added after a plausible observation cutoff."""
    return pd.DataFrame(
        {
            schema.CVE_ID: pd.Series(
                ["CVE-2020-1000", "CVE-2020-1004", "CVE-2020-1011"], dtype="string"
            ),
            schema.LABEL_DATE: pd.to_datetime(["2021-11-03", "2023-02-14", "2026-01-05"]),
        }
    )


@pytest.fixture
def imbalanced_labels() -> tuple[np.ndarray, np.ndarray]:
    """1000 items at a 1% positive rate, with scores that rank positives well."""
    rng = np.random.default_rng(4266)
    n = 1000
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=10, replace=False)] = 1
    scores = rng.random(n) * 0.5 + y * 0.4
    return y, scores
