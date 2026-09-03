# Data dictionary

Column contract for the analysis table. The authoritative version is
`src/vulnprio/data/schema.py` — this file is the human-readable companion.

## Identity and dates

| Column | Type | Source | Notes |
|---|---|---|---|
| `cve_id` | string | NVD | `CVE-YYYY-NNNN`. Primary key. |
| `published_date` | datetime | NVD | Drives the temporal split. |
| `last_modified_date` | datetime | NVD | Post-publication edits. Partial leak risk. |
| `description` | string | NVD | English text. **Leak risk — see `leakage.py`.** |

## CVSS v3.1

| Column | Type | Values |
|---|---|---|
| `cvss_base_score` | float | 0.0–10.0, discrete. **Heavily tied at 9.8.** |
| `cvss_vector` | string | e.g. `CVSS:3.1/AV:N/AC:L/...` |
| `cvss_severity` | string | LOW / MEDIUM / HIGH / CRITICAL |
| `attack_vector` | string | NETWORK, ADJACENT_NETWORK, LOCAL, PHYSICAL |
| `attack_complexity` | string | LOW, HIGH |
| `privileges_required` | string | NONE, LOW, HIGH |
| `user_interaction` | string | NONE, REQUIRED |
| `scope` | string | UNCHANGED, CHANGED |
| `confidentiality_impact` | string | NONE, LOW, HIGH |
| `integrity_impact` | string | NONE, LOW, HIGH |
| `availability_impact` | string | NONE, LOW, HIGH |

## Classification and software

| Column | Type | Notes |
|---|---|---|
| `cwe_ids` | list[str] | May contain `NVD-CWE-noinfo` / `NVD-CWE-Other` sentinels. |
| `cpe_vendors` | list[str] | Parsed from CPE URIs, not string-matched. |
| `cpe_products` | list[str] | Same. |
| `reference_urls` | list[str] | Accumulates post-publication. **Leak risk.** |
| `reference_tags` | list[str] | NVD-assigned. `Exploit` is the risky one. |

## Label and split

| Column | Type | Notes |
|---|---|---|
| `exploited` | int8 | Target. 1 iff in KEV with `dateAdded <= observation_cutoff`. |
| `kev_date_added` | datetime | NaT for negatives. **Never a feature.** |
| `split` | string | `train` / `valid` / `test` / `excluded`. |

## EPSS (joined for the baseline only)

| Column | Type | Notes |
|---|---|---|
| `epss_score` | float | 0–1 probability of exploitation in 30 days. Must be as-of a valid date. |
| `epss_percentile` | float | Rank within all scored CVEs on that date. |
