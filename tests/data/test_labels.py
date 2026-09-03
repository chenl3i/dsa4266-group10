"""Label construction tests."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from vulnprio.data import schema
from vulnprio.data.labels import attach_labels, label_summary


def test_kev_membership_becomes_the_positive_class(tiny_cves, tiny_kev):
    out = attach_labels(tiny_cves, tiny_kev, observation_cutoff=dt.date(2025, 6, 30))
    positives = set(out.loc[out[schema.LABEL] == 1, schema.CVE_ID])
    assert positives == {"CVE-2020-1000", "CVE-2020-1004"}


def test_entries_added_after_the_cutoff_are_not_positives(tiny_cves, tiny_kev):
    # CVE-2020-1011 was KEV-listed in 2026, past the cutoff. It must stay
    # negative, so that re-running against a fresher KEV snapshot does not
    # silently rewrite historical results.
    out = attach_labels(tiny_cves, tiny_kev, observation_cutoff=dt.date(2025, 6, 30))
    row = out.loc[out[schema.CVE_ID] == "CVE-2020-1011"].iloc[0]
    assert row[schema.LABEL] == 0
    assert pd.isna(row[schema.LABEL_DATE])


def test_earlier_cutoff_yields_fewer_positives(tiny_cves, tiny_kev):
    early = attach_labels(tiny_cves, tiny_kev, observation_cutoff=dt.date(2022, 1, 1))
    late = attach_labels(tiny_cves, tiny_kev, observation_cutoff=dt.date(2025, 6, 30))
    assert early[schema.LABEL].sum() < late[schema.LABEL].sum()


def test_row_count_is_preserved(tiny_cves, tiny_kev):
    out = attach_labels(tiny_cves, tiny_kev, observation_cutoff=dt.date(2025, 6, 30))
    assert len(out) == len(tiny_cves)


def test_duplicate_kev_entries_do_not_inflate_the_count(tiny_cves, tiny_kev):
    duped = pd.concat([tiny_kev, tiny_kev.iloc[[0]]], ignore_index=True)
    out = attach_labels(tiny_cves, duped, observation_cutoff=dt.date(2025, 6, 30))
    assert len(out) == len(tiny_cves)
    assert out[schema.LABEL].sum() == 2


def test_missing_cve_id_column_raises(tiny_kev):
    with pytest.raises(schema.SchemaError):
        attach_labels(
            pd.DataFrame({"wrong": [1]}), tiny_kev, observation_cutoff=dt.date(2025, 6, 30)
        )


def test_label_summary_reports_imbalance(tiny_cves, tiny_kev):
    out = attach_labels(tiny_cves, tiny_kev, observation_cutoff=dt.date(2025, 6, 30))
    summary = label_summary(out)
    assert summary["n_cves"] == 12
    assert summary["n_positive"] == 2
    assert summary["positive_rate"] == pytest.approx(2 / 12)
    assert summary["imbalance_ratio"] == pytest.approx(5.0)
