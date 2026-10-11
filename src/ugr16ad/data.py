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
# The proposal does not fix this split. 0.5 is a placeholder for the team to confirm.
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
    raise NotImplementedError("Phone, week 1: see docs/team-plan.md")


def segment_ids(index: pd.DatetimeIndex) -> pd.Series:
    """Number each run of consecutive minutes 0, 1, 2, ...

    A new segment starts wherever the step to the next timestamp is longer
    than one minute (the DST hour on 2016-03-27, the 1-6 April outage).
    Gaps are never filled, so this is how windows learn where they must stop.
    Returns an int Series on ``index``.
    """
    raise NotImplementedError("Phone, week 1: see docs/team-plan.md")


def make_part(X: pd.DataFrame, Y: pd.DataFrame) -> Part:
    """Build a Part from aligned X and Y frames (output of ``read_split``, or slices of it).

    ``labels`` keeps the 7 ``ATTACK_LABELS`` columns in that order and drops
    ``BLACKLIST_LABEL``. ``segment`` is computed on this part's own index, so it
    restarts at 0.
    """
    raise NotImplementedError("Phone, week 1: see docs/team-plan.md")


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
    raise NotImplementedError("Phone, week 1: see docs/team-plan.md")
