"""Time-based train/validation/test splitting.

A random split would let the model learn from CVEs published *after* the ones it
is scored on, which is not the decision the project is modelling. The real task
is: standing at a point in time, rank the vulnerabilities known so far.

Two subtleties the brief's Step 4 does not yet cover:

**Right-censoring.** Exploitation is observed with a lag. A CVE published a week
before ``train_end`` looks unexploited mostly because nobody has had time to
exploit it - or for CISA to notice. ``embargo_days`` drops that trailing window
so the training set is not taught that "recent" implies "safe".

**Label horizon.** Labels are still collected past the test boundary (up to
``labels.observation_cutoff``), because a test CVE genuinely exploited six months
after publication is a true positive. Restricting label collection to the split
boundary would fabricate a "recent CVEs are never exploited" pattern.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from vulnprio.data import schema

TRAIN = "train"
VALID = "valid"
TEST = "test"
EXCLUDED = "excluded"


@dataclass(frozen=True)
class TemporalSplit:
    train_end: dt.date
    valid_end: dt.date
    test_end: dt.date
    embargo_days: int = 0

    def assign(self, frame: pd.DataFrame) -> pd.Series:
        """Return a ``split`` label per row, based on ``published_date``.

        Rows published after ``test_end``, or inside the embargo window that ends
        at ``train_end``, are marked ``excluded`` rather than dropped, so the
        count of what was set aside stays visible in the report.
        """
        schema.validate(frame, {schema.PUBLISHED: "datetime64[ns]"}, name="frame")
        published = pd.to_datetime(frame[schema.PUBLISHED])

        train_end = pd.Timestamp(self.train_end)
        embargo_start = train_end - pd.Timedelta(days=self.embargo_days)

        split = pd.Series(EXCLUDED, index=frame.index, dtype="object")
        split[published <= embargo_start] = TRAIN
        split[(published > train_end) & (published <= pd.Timestamp(self.valid_end))] = VALID
        split[
            (published > pd.Timestamp(self.valid_end)) & (published <= pd.Timestamp(self.test_end))
        ] = TEST
        return split

    def describe(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-split row counts, positive counts and positive rates."""
        split = self.assign(frame)
        out = (
            frame.assign(**{schema.SPLIT: split})
            .groupby(schema.SPLIT)[schema.LABEL]
            .agg(n="size", n_positive="sum")
        )
        out["positive_rate"] = out["n_positive"] / out["n"]
        return out.reindex([TRAIN, VALID, TEST, EXCLUDED]).dropna(how="all")
