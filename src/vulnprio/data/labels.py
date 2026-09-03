"""Target construction: ``exploited in {0, 1}``.

The brief defines the target as KEV membership. That is right, but it hides an
assumption worth making explicit in the report: KEV lists *known* exploited
vulnerabilities. A CVE absent from KEV may be unexploited, or exploited and
unobserved. The negative class is therefore an *unlabelled* class, which makes
this positive-unlabelled (PU) learning, not clean binary classification.

We still train on it as binary - that is the standard, defensible choice - but
the consequence is stated in the report: precision@k is a *lower bound* on true
precision, because some scored-high "false positives" may be genuine exploits
that never made the catalogue.

See docs/decisions/0002-label-definition.md.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from vulnprio.data import schema


def attach_labels(
    cves: pd.DataFrame,
    kev: pd.DataFrame,
    *,
    observation_cutoff: dt.date,
) -> pd.DataFrame:
    """Join the KEV catalogue onto the CVE table and derive the binary target.

    Args:
        cves: One row per CVE, carrying at least ``cve_id``.
        kev: One row per KEV entry, carrying ``cve_id`` and ``kev_date_added``.
        observation_cutoff: Labels are collected as of this date. A KEV entry
            added *after* it is ignored, so that re-running the pipeline later
            against a fresher KEV snapshot does not silently change historical
            results.

    Returns:
        ``cves`` with ``exploited`` (int8) and ``kev_date_added`` (NaT when the
        CVE is not a positive) appended.
    """
    schema.validate(cves, {schema.CVE_ID: "string"}, name="cves")
    schema.validate(kev, {schema.CVE_ID: "string", schema.LABEL_DATE: "datetime64[ns]"}, name="kev")

    cutoff = pd.Timestamp(observation_cutoff)
    in_window = kev.loc[kev[schema.LABEL_DATE] <= cutoff, [schema.CVE_ID, schema.LABEL_DATE]]
    in_window = in_window.drop_duplicates(subset=schema.CVE_ID, keep="first")

    labelled = cves.merge(in_window, on=schema.CVE_ID, how="left", validate="one_to_one")
    labelled[schema.LABEL] = labelled[schema.LABEL_DATE].notna().astype("int8")
    return labelled


def label_summary(labelled: pd.DataFrame) -> dict[str, float]:
    """Class balance figures for the report.

    The brief flags this: NVD holds 200k+ CVEs against roughly 1,300 KEV
    entries, so the positive rate sits near 0.5%. Any metric that can be gamed
    by predicting all-negative (accuracy, and to a lesser extent F1 at a fixed
    threshold) is meaningless here.
    """
    n = len(labelled)
    positives = int(labelled[schema.LABEL].sum())
    return {
        "n_cves": n,
        "n_positive": positives,
        "n_negative": n - positives,
        "positive_rate": positives / n if n else 0.0,
        "imbalance_ratio": (n - positives) / positives if positives else float("inf"),
    }
