"""Load UGR'16 v2 and cut it into the four chronological parts every model shares.

One loader for everyone is what makes the model comparison fair: if each model
read the CSVs its own way, a difference in results could come from the loading,
not the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ugr16ad.paths import UGR16_DIR

KEY = "Row"
ATTACK_LABELS = (
    "labeldos",
    "labelscan11",
    "labelscan44",
    "labelnerisbotnet",
    "labelanomalyidpscan",
    "labelanomalysshscan",
    "labelanomalyspam",
)
# Non-zero in every minute of both files: it marks IP reputation, not an attack.
BLACKLIST_LABEL = "labelblacklist"
# Proposal: the first 78,609 training rows fit the models; the other 19,653 validate.
FIT_ROWS = 78_609
# Share of validation rows used for tuning; the rest calibrate the threshold.
# Settled by the team on 26 Oct: each half is ~6.8 days, close to a full weekly
# cycle, so early stopping and the threshold both see weekday and weekend traffic.
TUNE_FRACTION = 0.5


@dataclass(frozen=True)
class Part:
    """One chronological slice of the data. All fields share one DatetimeIndex.

    Attributes:
        X: the 134 feature counts. Never contains ``Row`` or any label.
        y: True when any of the 7 attack labels is > 0 in that minute.
        labels: the 7 attack label counts, for per-category recall.
        segment: run id of contiguous minutes, restarting at 0 in every part.
            Windows must stay inside one segment.
    """

    X: pd.DataFrame
    y: pd.Series
    labels: pd.DataFrame
    segment: pd.Series


def read_split(
    split: Literal["train", "test"], directory: Path = UGR16_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``UGR16v2.X<split>.csv`` and ``UGR16v2.Y<split>.csv``.

    Both frames come back indexed by a DatetimeIndex named ``"timestamp"``,
    parsed from ``Row`` with format ``%Y%m%d%H%M``, and ``Row`` is dropped.
    The checks hold for today's files, but asserting them means a re-exported
    copy cannot silently pair the wrong label with a minute.

    Raises:
        ValueError: if X and Y keys differ, keys are not strictly increasing,
            or any key repeats.
    """
    X = pd.read_csv(directory / f"UGR16v2.X{split}.csv")
    Y = pd.read_csv(directory / f"UGR16v2.Y{split}.csv")
    if not X[KEY].equals(Y[KEY]):
        raise ValueError(f"{split}: X and Y {KEY} keys differ")
    if X[KEY].duplicated().any():
        raise ValueError(f"{split}: {KEY} keys repeat")
    if not X[KEY].is_monotonic_increasing:
        raise ValueError(f"{split}: {KEY} keys are not increasing")

    index = pd.DatetimeIndex(
        pd.to_datetime(X[KEY].astype(str), format="%Y%m%d%H%M"), name="timestamp"
    )
    X = X.drop(columns=KEY).set_index(index)
    Y = Y.drop(columns=KEY).set_index(index)
    return X, Y


def segment_ids(index: pd.DatetimeIndex) -> pd.Series:
    """Number each run of consecutive minutes 0, 1, 2, ...

    A new segment starts wherever the step to the next timestamp is longer
    than one minute (the DST hour on 2016-03-27, the 1-6 April outage).
    Gaps are never filled, so this is how windows learn where they must stop.
    Returns an int Series on ``index``.
    """
    # The first step is NaT, which counts as a break: ids start at 1, so shift to 0.
    breaks = index.to_series().diff() != pd.Timedelta(minutes=1)
    return (breaks.cumsum() - 1).astype("int64")


def make_part(X: pd.DataFrame, Y: pd.DataFrame) -> Part:
    """Build a Part from aligned X and Y frames (output of ``read_split``, or slices of it).

    ``labels`` keeps the 7 ``ATTACK_LABELS`` columns in that order and drops
    ``BLACKLIST_LABEL``. ``segment`` is computed on this part's own index, so it
    restarts at 0.
    """
    labels = Y[list(ATTACK_LABELS)]
    return Part(
        X=X,
        y=(labels > 0).any(axis=1),
        labels=labels,
        segment=segment_ids(X.index),
    )


def load_dataset(
    directory: Path = UGR16_DIR,
    fit_rows: int = FIT_ROWS,
    tune_fraction: float = TUNE_FRACTION,
) -> dict[str, Part]:
    """Return ``{"fit", "tune", "calib", "test"}`` Parts, in time order.

    The first ``fit_rows`` training rows are "fit". Of the remaining training
    rows, the first ``int(len(rest) * tune_fraction)`` are "tune" and the rest
    are "calib". "test" is the whole test file. Nothing is shuffled: a later
    part always starts after an earlier one ends.
    """
    X, Y = read_split("train", directory)
    n_tune = int((len(X) - fit_rows) * tune_fraction)
    cuts = {
        "fit": slice(0, fit_rows),
        "tune": slice(fit_rows, fit_rows + n_tune),
        "calib": slice(fit_rows + n_tune, len(X)),
    }
    parts = {name: make_part(X.iloc[rows], Y.iloc[rows]) for name, rows in cuts.items()}
    parts["test"] = make_part(*read_split("test", directory))
    return parts
