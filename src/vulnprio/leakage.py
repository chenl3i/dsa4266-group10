"""Leakage guards.

The draft brief ends on "Data leakage risk". This module is that section made
executable. Four distinct leaks threaten this project, and the third is the one
that quietly inflates results in published work on this task.

1. **Temporal leakage.** Training on CVEs published after the test window.
   Prevented structurally by :mod:`vulnprio.data.splits`.

2. **Label-date leakage.** Using ``kev_date_added`` - or anything derived from
   it - as a feature. It exists only because the CVE is a positive.

3. **Description leakage.** NVD descriptions are edited over time. A CVE that
   gets exploited often has its description *rewritten* afterwards to say so.
   Training TF-IDF on text containing "exploited in the wild", "actively
   exploited" or "ransomware campaign" is training on the answer. The
   blocklist below is applied to the text vectoriser.

4. **Post-hoc reference leakage.** Reference URLs and tags accumulate after
   publication. An ``Exploit`` tag or a link to a Metasploit module may well be
   *why* the CVE reached KEV. Reference features are supported, but must be
   reported both with and without - the ablation is a finding, not an
   inconvenience.

None of these is fully solvable with NVD's current-state snapshots, which do not
expose historical field revisions. Documenting the residual risk honestly is
part of the deliverable.
"""

from __future__ import annotations

import re

import pandas as pd

from vulnprio.data import schema

#: Phrases whose presence in a description is plausibly a consequence of the
#: label rather than a predictor of it. Matched case-insensitively as whole
#: phrases and stripped before vectorisation.
LEAKY_PHRASES: tuple[str, ...] = (
    "exploited in the wild",
    "actively exploited",
    "exploitation detected",
    "known exploited",
    "in-the-wild exploitation",
    "has been exploited",
    "is being exploited",
    "was exploited",
    "ransomware",
    "zero-day",
    "zero day",
    "0-day",
    "cisa kev",
    "known to be exploited",
    "under active attack",
    "observed in attacks",
)

#: Columns that must never reach a model - they are label artefacts.
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {schema.LABEL, schema.LABEL_DATE, schema.SPLIT, "kev_notes", "kev_required_action"}
)

_LEAK_PATTERN = re.compile("|".join(re.escape(p) for p in LEAKY_PHRASES), flags=re.IGNORECASE)


def scrub_text(text: str) -> str:
    """Remove leaky phrases from a description, collapsing the resulting gaps."""
    return re.sub(r"\s{2,}", " ", _LEAK_PATTERN.sub(" ", text)).strip()


def scrub_series(series: pd.Series) -> pd.Series:
    """Vectorised :func:`scrub_text` over a description column."""
    return series.fillna("").map(scrub_text)


class LeakageError(AssertionError):
    """Raised when a feature matrix carries a column it must not."""


def assert_no_forbidden_columns(frame: pd.DataFrame) -> None:
    """Fail loudly if a label artefact reached the feature matrix."""
    offenders = sorted(set(frame.columns) & FORBIDDEN_FEATURE_COLUMNS)
    if offenders:
        raise LeakageError(
            f"Label-derived columns present in feature matrix: {offenders}. "
            "These exist only because the label exists and must be dropped."
        )


def assert_temporal_order(
    train: pd.DataFrame, test: pd.DataFrame, *, column: str = schema.PUBLISHED
) -> None:
    """Fail loudly if any training record is newer than the oldest test record."""
    if train.empty or test.empty:
        return
    latest_train = pd.to_datetime(train[column]).max()
    earliest_test = pd.to_datetime(test[column]).min()
    if latest_train >= earliest_test:
        raise LeakageError(
            f"Temporal leakage: newest training record ({latest_train:%Y-%m-%d}) is not "
            f"strictly older than the oldest test record ({earliest_test:%Y-%m-%d})."
        )


def leakage_audit(frame: pd.DataFrame) -> dict[str, int]:
    """Count how many descriptions contain leaky phrases - a number for the report."""
    if schema.DESCRIPTION not in frame.columns:
        return {}
    text = frame[schema.DESCRIPTION].fillna("")
    counts = {
        phrase: int(text.str.contains(re.escape(phrase), case=False).sum())
        for phrase in LEAKY_PHRASES
    }
    return {k: v for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v}
