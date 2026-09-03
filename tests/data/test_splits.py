"""Temporal split tests.

The property that matters most is the last one: no training record may be newer
than any test record. Everything else is bookkeeping around that guarantee.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from vulnprio.data import schema
from vulnprio.data.splits import EXCLUDED, TEST, TRAIN, VALID, TemporalSplit

SPLIT = TemporalSplit(
    train_end=dt.date(2022, 12, 31),
    valid_end=dt.date(2023, 6, 30),
    test_end=dt.date(2023, 12, 31),
    embargo_days=0,
)


def test_assigns_by_publication_date(tiny_cves):
    labels = SPLIT.assign(tiny_cves)
    published = pd.to_datetime(tiny_cves[schema.PUBLISHED])
    assert (labels[published <= pd.Timestamp("2022-12-31")] == TRAIN).all()
    assert (labels[published > pd.Timestamp("2023-12-31")] == EXCLUDED).all()


def test_embargo_moves_recent_training_rows_to_excluded(tiny_cves):
    embargoed = TemporalSplit(
        train_end=dt.date(2022, 12, 31),
        valid_end=dt.date(2023, 6, 30),
        test_end=dt.date(2023, 12, 31),
        embargo_days=180,
    )
    plain_train = (SPLIT.assign(tiny_cves) == TRAIN).sum()
    embargo_train = (embargoed.assign(tiny_cves) == TRAIN).sum()
    assert embargo_train < plain_train
    # Embargoed rows are set aside, never silently deleted.
    assert len(embargoed.assign(tiny_cves)) == len(tiny_cves)


def test_no_training_record_is_newer_than_any_test_record(tiny_cves):
    labels = SPLIT.assign(tiny_cves)
    published = pd.to_datetime(tiny_cves[schema.PUBLISHED])
    train_max = published[labels == TRAIN].max()
    test_min = published[labels == TEST].min()
    assert train_max < test_min


def test_every_row_receives_exactly_one_split(tiny_cves):
    labels = SPLIT.assign(tiny_cves)
    assert len(labels) == len(tiny_cves)
    assert set(labels.unique()) <= {TRAIN, VALID, TEST, EXCLUDED}
    assert labels.notna().all()
