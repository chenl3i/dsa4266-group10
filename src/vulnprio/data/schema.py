"""The canonical column contract for the analysis table.

Every downstream module reads these names, so a change to the NVD parser cannot
silently break feature code - the schema check fails first. Keeping the contract
in one place is what lets feature blocks be written independently of each other.
"""

from __future__ import annotations

import pandas as pd

CVE_ID = "cve_id"
PUBLISHED = "published_date"
LAST_MODIFIED = "last_modified_date"
DESCRIPTION = "description"

CVSS_BASE_SCORE = "cvss_base_score"
CVSS_VECTOR = "cvss_vector"
CVSS_SEVERITY = "cvss_severity"
CVSS_ATTACK_VECTOR = "attack_vector"
CVSS_ATTACK_COMPLEXITY = "attack_complexity"
CVSS_PRIVILEGES_REQUIRED = "privileges_required"
CVSS_USER_INTERACTION = "user_interaction"
CVSS_SCOPE = "scope"
CVSS_CONF_IMPACT = "confidentiality_impact"
CVSS_INTEG_IMPACT = "integrity_impact"
CVSS_AVAIL_IMPACT = "availability_impact"

CWE_IDS = "cwe_ids"
CPE_VENDORS = "cpe_vendors"
CPE_PRODUCTS = "cpe_products"
REFERENCE_URLS = "reference_urls"
REFERENCE_TAGS = "reference_tags"

LABEL = "exploited"
LABEL_DATE = "kev_date_added"
SPLIT = "split"

#: Columns the NVD parser must produce.
NVD_COLUMNS: dict[str, str] = {
    CVE_ID: "string",
    PUBLISHED: "datetime64[ns]",
    LAST_MODIFIED: "datetime64[ns]",
    DESCRIPTION: "string",
    CVSS_BASE_SCORE: "float64",
    CVSS_VECTOR: "string",
    CVSS_SEVERITY: "string",
    CVSS_ATTACK_VECTOR: "string",
    CVSS_ATTACK_COMPLEXITY: "string",
    CVSS_PRIVILEGES_REQUIRED: "string",
    CVSS_USER_INTERACTION: "string",
    CVSS_SCOPE: "string",
    CVSS_CONF_IMPACT: "string",
    CVSS_INTEG_IMPACT: "string",
    CVSS_AVAIL_IMPACT: "string",
    CWE_IDS: "object",
    CPE_VENDORS: "object",
    CPE_PRODUCTS: "object",
    REFERENCE_URLS: "object",
    REFERENCE_TAGS: "object",
}

#: Columns added by the labelling step.
LABEL_COLUMNS: dict[str, str] = {LABEL: "int8", LABEL_DATE: "datetime64[ns]"}

#: The eight CVSS v3.1 base-metric categoricals, with their permitted values.
CVSS_CATEGORICALS: dict[str, tuple[str, ...]] = {
    CVSS_ATTACK_VECTOR: ("NETWORK", "ADJACENT_NETWORK", "LOCAL", "PHYSICAL"),
    CVSS_ATTACK_COMPLEXITY: ("LOW", "HIGH"),
    CVSS_PRIVILEGES_REQUIRED: ("NONE", "LOW", "HIGH"),
    CVSS_USER_INTERACTION: ("NONE", "REQUIRED"),
    CVSS_SCOPE: ("UNCHANGED", "CHANGED"),
    CVSS_CONF_IMPACT: ("NONE", "LOW", "HIGH"),
    CVSS_INTEG_IMPACT: ("NONE", "LOW", "HIGH"),
    CVSS_AVAIL_IMPACT: ("NONE", "LOW", "HIGH"),
}


class SchemaError(ValueError):
    """Raised when a DataFrame violates the declared column contract."""


def validate(frame: pd.DataFrame, expected: dict[str, str], *, name: str = "frame") -> None:
    """Assert that ``frame`` carries every column in ``expected``.

    Checks presence, not dtype equality - pandas dtype inference is too loose for
    an equality check to be useful, but a missing column is always a real bug.
    """
    missing = [col for col in expected if col not in frame.columns]
    if missing:
        raise SchemaError(f"{name} is missing required columns: {missing}")
