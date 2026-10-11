# Week 1 Starter Files Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push starter files (full signatures, docstrings, `NotImplementedError` bodies, and tests that define "done") so each of the four members can build their part of the common pipeline in week 1 without waiting on anyone.

**Architecture:** Flat modules in `src/ugr16ad/`, one owner each. Every test that needs an unimplemented stub is marked `@todo` (a strict `xfail` on `NotImplementedError`), so CI stays green now. When a member implements a stub, its tests either pass (and strict xfail forces them to delete `@todo`) or fail on a real assertion. Each task's tests are checked against a throwaway reference implementation before the stub is restored, so members never chase a wrong test.

**Tech Stack:** Python ≥3.10, pandas, numpy, scikit-learn, PyTorch (CPU in CI), pytest, ruff.

**Spec:** `docs/DSA4266.md` (the proposal) plus the week-by-week split agreed in chat on 2026-10-11 (recorded in `docs/team-plan.md`, Task 6).

## Global Constraints

- Owners: Phone = data loading (`data.py`, `synthetic.py`); Kesha = input prep (`preprocess.py`, `windows.py`); Vanessa = scoring (`evaluate.py`, `results.py`); Chen Lei = training tools + PCA (`train.py`, `models/pca.py`).
- Stub bodies are exactly `raise NotImplementedError("<Owner>, week 1: see docs/team-plan.md")`.
- Docstrings explain **why**, not what (repo convention, CLAUDE.md).
- Train rows `0..78_608` fit models; rows `78_609..98_261` validate (tune block first, calibration block second). No split ever shuffles rows.
- Preprocessing parameters are fitted on the fit part only.
- Threshold = 99th percentile of calibration scores; an alert is `score > threshold`.
- `Row` and `labelblacklist` are never model inputs; the target is any of the 7 non-blacklist labels > 0.
- Windows end at minute t, contain only t and earlier minutes, never cross a gap or a part boundary; the detection score is the final minute's mean squared reconstruction error.
- Every scorer (PCA and autoencoders) returns `(scores: (n,), feature_errors: (n, f))`.
- Run results go to `reports/results/<model>/seed<seed>/` as `scores.csv`, `meta.json`, optional `latents.npy`, optional `feature_errors.npy`.
- Tests use tiny synthetic data; only one `slow` test reads the real CSVs.
- `pytest` and `ruff check src tests` + `ruff format --check src tests` must pass after every task.

## Review Focus

1. **A member implements a stub but leaves `@todo` on.** Strict xfail turns the XPASS into a failure that says "delete @todo". Pinned by the `todo` marker definition (Task 1).
2. **Train and test parts are separate files, but `fit`/`tune`/`calib` are slices of one file.** A window in `tune` must not reach back into `fit`. Pinned by `test_segment_restarts_per_part` (Task 2) together with `test_windows_never_cross_segments` (Task 3).
3. **Gaps in time inside a part (DST hour, April outage).** Segment ids must change at every gap longer than 1 minute. Pinned by `test_segment_ids_increment_at_gaps` (Task 2).
4. **A threshold run with zero alerts on test.** Precision and F1 must be 0.0, not a crash or a NaN. Pinned by `test_threshold_metrics_with_no_alerts` (Task 4).
5. **Attack categories absent from test (sshscan, spam).** Recall is NaN with `n_positive == 0`, not a divide-by-zero. Pinned by `test_per_category_recall_absent_category_is_nan` (Task 4).

---

### Task 1: Test harness and PyTorch dependency

**Files:**
- Modify: `requirements.txt`
- Modify: `.github/workflows/ci.yml`
- Create: `tests/stubs.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `from stubs import todo` (pytest marker) and the fixture `raw_dir(tmp_path) -> Path`, a folder holding tiny `UGR16v2.{X,Y}{train,test}.csv` files.

- [ ] **Step 1: Add torch to `requirements.txt`**

Insert under `# Modelling`:

```text
# Modelling
scikit-learn>=1.4
torch>=2.4
```

- [ ] **Step 2: Install CPU torch in CI before the rest**

In `.github/workflows/ci.yml`, replace the Install step's `run` block with:

```yaml
        run: |
          python -m pip install --upgrade pip
          # CPU wheel: the default Linux wheel pulls ~2 GB of CUDA libraries CI never uses.
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt
          pip install -e .
```

- [ ] **Step 3: Install torch locally**

Run: `./.venv/Scripts/python.exe -m pip install -r requirements.txt`
Expected: torch installs; `./.venv/Scripts/python.exe -c "import torch; print(torch.__version__)"` prints a version.

- [ ] **Step 4: Create `tests/stubs.py`**

```python
"""The marker for tests whose code is still a starter stub.

Strict xfail on NotImplementedError does three jobs at once: CI stays green
while stubs are empty, a wrong implementation fails on its real assertion, and
a correct one fails as XPASS until its owner deletes the marker, so finished
work cannot hide behind it.
"""

import pytest

todo = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="starter stub not implemented yet; once it passes, delete @todo",
)
```

- [ ] **Step 5: Create `tests/conftest.py`**

```python
from pathlib import Path

import pandas as pd
import pytest

LABEL_COLUMNS = [
    "labeldos",
    "labelscan11",
    "labelscan44",
    "labelnerisbotnet",
    "labelblacklist",
    "labelanomalyidpscan",
    "labelanomalysshscan",
    "labelanomalyspam",
]


def _minutes(start: str, n: int) -> list[int]:
    stamps = pd.date_range(start, periods=n, freq="min")
    return [int(t.strftime("%Y%m%d%H%M")) for t in stamps]


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Tiny stand-in for data/external/ugr16v2/, same file names and columns.

    Train: 6 contiguous minutes, a 4-minute gap, then 4 more (10 rows, no attacks).
    Test: 5 contiguous minutes; row 2 is DoS and row 4 is spam.
    Blacklist is non-zero everywhere, as in the real data.
    """
    train_rows = _minutes("2016-03-19 00:00", 6) + _minutes("2016-03-19 00:10", 4)
    test_rows = _minutes("2016-07-28 00:00", 5)
    for split, rows in (("train", train_rows), ("test", test_rows)):
        n = len(rows)
        X = pd.DataFrame(
            {
                "Row": rows,
                "f_a": range(1, n + 1),
                "f_b": [i % 3 for i in range(n)],
                "f_const": [5] * n,
            }
        )
        Y = pd.DataFrame({"Row": rows, **{c: [0] * n for c in LABEL_COLUMNS}})
        Y["labelblacklist"] = 7
        if split == "test":
            Y.loc[2, "labeldos"] = 3
            Y.loc[4, "labelanomalyspam"] = 1
        X.to_csv(tmp_path / f"UGR16v2.X{split}.csv", index=False)
        Y.to_csv(tmp_path / f"UGR16v2.Y{split}.csv", index=False)
    return tmp_path
```

- [ ] **Step 6: Verify nothing broke**

Run: `./.venv/Scripts/python.exe -m pytest` → Expected: `3 passed`.
Run: `./.venv/Scripts/python.exe -m ruff check src tests` and `... ruff format --check src tests` → Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .github/workflows/ci.yml tests/stubs.py tests/conftest.py
git commit -m "Add torch dependency, @todo marker and tiny CSV fixture for starter files"
```

---

### Task 2: Phone — data loading (`data.py`, `synthetic.py`)

**Files:**
- Create: `src/ugr16ad/data.py`
- Create: `src/ugr16ad/synthetic.py`
- Test: `tests/test_data.py`, `tests/test_synthetic.py`

**Interfaces:**
- Consumes: `ugr16ad.paths.UGR16_DIR`; fixture `raw_dir`; marker `todo`.
- Produces:
  - `KEY: str`, `ATTACK_LABELS: tuple[str, ...]` (7), `BLACKLIST_LABEL: str`, `FIT_ROWS: int = 78_609`, `TUNE_FRACTION: float = 0.5`
  - `Part` frozen dataclass: `X: pd.DataFrame`, `y: pd.Series[bool]`, `labels: pd.DataFrame` (7 attack columns), `segment: pd.Series[int]`; all share one `DatetimeIndex` named `"timestamp"`.
  - `read_split(split: Literal["train", "test"], directory: Path = UGR16_DIR) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `segment_ids(index: pd.DatetimeIndex) -> pd.Series`
  - `make_part(X: pd.DataFrame, Y: pd.DataFrame) -> Part`
  - `load_dataset(directory: Path = UGR16_DIR, fit_rows: int = FIT_ROWS, tune_fraction: float = TUNE_FRACTION) -> dict[str, Part]` with keys `"fit", "tune", "calib", "test"` in that order.
  - `inject_anomalies(X: np.ndarray, *, rate: float, magnitude: float, n_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write `tests/test_data.py`**

```python
import pandas as pd
import pytest
from stubs import todo

from ugr16ad import data


def test_attack_labels_exclude_blacklist():
    assert len(data.ATTACK_LABELS) == 7
    assert data.BLACKLIST_LABEL not in data.ATTACK_LABELS


@todo
def test_read_split_indexes_by_timestamp_and_drops_key(raw_dir):
    X, Y = data.read_split("train", directory=raw_dir)
    assert data.KEY not in X.columns
    assert data.KEY not in Y.columns
    assert X.index.name == "timestamp"
    assert X.index[0] == pd.Timestamp("2016-03-19 00:00")
    assert X.index.equals(Y.index)
    assert list(X.columns) == ["f_a", "f_b", "f_const"]


@todo
def test_read_split_rejects_misaligned_keys(raw_dir):
    path = raw_dir / "UGR16v2.Ytrain.csv"
    Y = pd.read_csv(path)
    Y.loc[0, "Row"] = 201601010000
    Y.to_csv(path, index=False)
    with pytest.raises(ValueError):
        data.read_split("train", directory=raw_dir)


@todo
def test_read_split_rejects_duplicate_keys(raw_dir):
    for prefix in ("X", "Y"):
        path = raw_dir / f"UGR16v2.{prefix}train.csv"
        frame = pd.read_csv(path)
        pd.concat([frame, frame.tail(1)]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        data.read_split("train", directory=raw_dir)


@todo
def test_segment_ids_increment_at_gaps():
    index = pd.DatetimeIndex(
        ["2016-03-19 00:00", "2016-03-19 00:01", "2016-03-19 00:02",
         "2016-03-19 00:10", "2016-03-19 00:11"]
    )
    assert data.segment_ids(index).tolist() == [0, 0, 0, 1, 1]


@todo
def test_make_part_target_ignores_blacklist(raw_dir):
    part = data.make_part(*data.read_split("test", directory=raw_dir))
    assert part.y.dtype == bool
    assert part.y.tolist() == [False, False, True, False, True]
    assert data.BLACKLIST_LABEL not in part.labels.columns
    assert list(part.labels.columns) == list(data.ATTACK_LABELS)
    assert part.segment.tolist() == [0, 0, 0, 0, 0]


@todo
def test_load_dataset_splits_chronologically(raw_dir):
    parts = data.load_dataset(directory=raw_dir, fit_rows=6, tune_fraction=0.5)
    assert list(parts) == ["fit", "tune", "calib", "test"]
    assert [len(p.X) for p in parts.values()] == [6, 2, 2, 5]
    assert parts["fit"].X.index.max() < parts["tune"].X.index.min()
    assert parts["tune"].X.index.max() < parts["calib"].X.index.min()


@todo
def test_segment_restarts_per_part(raw_dir):
    # Parts are windowed separately, so a window in tune can never reach into fit.
    parts = data.load_dataset(directory=raw_dir, fit_rows=6, tune_fraction=0.5)
    for part in parts.values():
        assert part.segment.iloc[0] == 0


@todo
@pytest.mark.slow
def test_real_data_matches_proposal():
    parts = data.load_dataset()
    assert len(parts["fit"].X) == 78_609
    assert len(parts["tune"].X) + len(parts["calib"].X) == 19_653
    assert len(parts["test"].X) == 43_200
    assert parts["fit"].X.shape[1] == 134
    assert not any(parts[name].y.any() for name in ("fit", "tune", "calib"))
    assert int(parts["test"].y.sum()) == 1_364
```

- [ ] **Step 2: Write `tests/test_synthetic.py`**

```python
import numpy as np
from stubs import todo

from ugr16ad.synthetic import inject_anomalies


@todo
def test_inject_anomalies_shifts_only_marked_rows():
    X = np.zeros((1000, 20), dtype=np.float32)
    injected, mask = inject_anomalies(X, rate=0.05, magnitude=4.0, n_features=3, seed=0)
    assert mask.dtype == bool
    assert mask.sum() == 50
    assert np.all(injected[~mask] == 0)
    assert np.all((injected[mask] != 0).sum(axis=1) == 3)
    assert np.allclose(injected[mask][injected[mask] != 0], 4.0)
    assert np.all(X == 0), "input must not be modified in place"


@todo
def test_inject_anomalies_is_reproducible():
    X = np.zeros((200, 10), dtype=np.float32)
    a, mask_a = inject_anomalies(X, rate=0.1, magnitude=3.0, n_features=2, seed=1)
    b, mask_b = inject_anomalies(X, rate=0.1, magnitude=3.0, n_features=2, seed=1)
    c, _ = inject_anomalies(X, rate=0.1, magnitude=3.0, n_features=2, seed=2)
    assert np.array_equal(a, b) and np.array_equal(mask_a, mask_b)
    assert not np.array_equal(a, c)
```

- [ ] **Step 3: Write the stub `src/ugr16ad/data.py`**

```python
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
```

- [ ] **Step 4: Write the stub `src/ugr16ad/synthetic.py`**

```python
"""Fake anomalies for the validation sanity check.

Validation has no labelled attacks, so this is the only way to check before
test that a model reacts to anything at all. Inputs are already standardised
with fit-part statistics, so a shift of ``magnitude`` means "this many fit-period
standard deviations" and uses no information from later data.
"""

from __future__ import annotations

import numpy as np


def inject_anomalies(
    X: np.ndarray, *, rate: float, magnitude: float, n_features: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(copy of X with anomalies, bool mask of altered rows)``.

    Picks ``round(rate * len(X))`` rows at random. In each, adds ``magnitude``
    to ``n_features`` randomly chosen columns (different columns per row).
    ``X`` is 2-D ``(rows, features)`` and is never modified in place. The same
    ``seed`` gives the same result.
    """
    raise NotImplementedError("Phone, week 1: see docs/team-plan.md")
```

- [ ] **Step 5: Verify the tests are correct with a throwaway reference implementation**

Temporarily replace each `raise NotImplementedError(...)` in both files with a real implementation, then run:
`./.venv/Scripts/python.exe -m pytest tests/test_data.py tests/test_synthetic.py --runxfail -v`
Expected: all PASS, including `test_real_data_matches_proposal`. Fix any **test** that is wrong (not the spec). Save the reference implementation to the session scratchpad (not the repo), then restore the stub bodies exactly as in Steps 3–4.

- [ ] **Step 6: Run with stubs**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_data.py tests/test_synthetic.py -v`
Expected: `1 passed`, everything else `XFAIL`. Then run `ruff check` and `ruff format --check` on `src tests`: clean.

- [ ] **Step 7: Commit**

```bash
git add src/ugr16ad/data.py src/ugr16ad/synthetic.py tests/test_data.py tests/test_synthetic.py
git commit -m "Add data loading and synthetic anomaly starter files (Phone)"
```

---

### Task 3: Kesha — input prep (`preprocess.py`, `windows.py`)

**Files:**
- Create: `src/ugr16ad/preprocess.py`
- Create: `src/ugr16ad/windows.py`
- Test: `tests/test_preprocess.py`, `tests/test_windows.py`

**Interfaces:**
- Consumes: nothing from other tasks at runtime. Callers pass `Part.X` (DataFrame) and `Part.segment.to_numpy()` (int array) from Task 2.
- Produces:
  - `class Preprocessor` with `fit(X: pd.DataFrame) -> Preprocessor`, `transform(X: pd.DataFrame) -> np.ndarray` (float32, `(n, k)`), attribute `feature_names_: list[str]` (kept columns, in input order).
  - `eligible_positions(segment: np.ndarray, length: int) -> np.ndarray` (int64 row positions)
  - `make_windows(X: np.ndarray, segment: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]` returning `(windows (m, length, f), positions (m,))`.

- [ ] **Step 1: Write `tests/test_preprocess.py`**

```python
import numpy as np
import pandas as pd
import pytest
from stubs import todo

from ugr16ad.preprocess import Preprocessor

FIT = pd.DataFrame({"a": [0, 1, 3, 7], "b": [2, 2, 2, 2], "c": [0, 0, 0, 100]})


@todo
def test_fit_drops_constant_columns_and_standardises():
    pre = Preprocessor().fit(FIT)
    assert pre.feature_names_ == ["a", "c"]
    Z = pre.transform(FIT)
    assert Z.dtype == np.float32
    assert Z.shape == (4, 2)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-6)


@todo
def test_transform_reuses_fit_statistics_unchanged():
    pre = Preprocessor().fit(FIT)
    later = pd.DataFrame({"a": [0], "b": [9], "c": [0]})
    logged = np.log1p(FIT["a"].to_numpy(dtype=float))
    expected_a = (np.log1p(0) - logged.mean()) / logged.std()
    np.testing.assert_allclose(pre.transform(later)[0, 0], expected_a, rtol=1e-5)


@todo
def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        Preprocessor().transform(FIT)


@todo
def test_transform_rejects_different_columns():
    pre = Preprocessor().fit(FIT)
    with pytest.raises(ValueError):
        pre.transform(FIT.drop(columns="a"))
```

- [ ] **Step 2: Write `tests/test_windows.py`**

```python
import numpy as np
from stubs import todo

from ugr16ad.windows import eligible_positions, make_windows

SEGMENT = np.array([0, 0, 0, 0, 1, 1, 1])
X = np.arange(7 * 2, dtype=np.float32).reshape(7, 2)


@todo
def test_eligible_positions_need_a_full_window_in_one_segment():
    assert eligible_positions(SEGMENT, 3).tolist() == [2, 3, 6]


@todo
def test_length_one_makes_every_row_eligible():
    assert eligible_positions(SEGMENT, 1).tolist() == list(range(7))


@todo
def test_windows_never_cross_segments():
    windows, positions = make_windows(X, SEGMENT, 3)
    assert windows.shape == (3, 3, 2)
    assert positions.tolist() == [2, 3, 6]
    np.testing.assert_array_equal(windows[0], X[0:3])
    np.testing.assert_array_equal(windows[2], X[4:7])
    np.testing.assert_array_equal(windows[:, -1, :], X[positions])


@todo
def test_window_longer_than_any_segment_gives_empty_result():
    windows, positions = make_windows(X, SEGMENT, 5)
    assert windows.shape == (0, 5, 2)
    assert positions.shape == (0,)
```

- [ ] **Step 3: Write the stub `src/ugr16ad/preprocess.py`**

```python
"""Turn raw counts into model inputs: log(1+x), drop constant columns, standardise.

Everything here is learned from the fit part only and then applied unchanged to
tune, calib and test. Fitting on later data would let the scaler peek at the
traffic the models are meant to be judged on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Preprocessor:
    """log1p -> drop columns constant in the fit part -> standard-scale.

    Mostly-zero columns are kept on purpose: a rarely used port can be exactly
    what an attack lights up. Standard scaling uses the population std
    (``ddof=0``), as scikit-learn's ``StandardScaler`` does.

    Attributes:
        feature_names_: the kept column names, in input order. Set by ``fit``.
    """

    feature_names_: list[str]

    def fit(self, X: pd.DataFrame) -> Preprocessor:
        """Learn the kept columns, and their log1p means and stds, from ``X``. Returns self."""
        raise NotImplementedError("Kesha, week 1: see docs/team-plan.md")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted steps. Returns float32 ``(rows, len(feature_names_))``.

        Raises:
            RuntimeError: if called before ``fit``.
            ValueError: if ``X`` lacks a column that was present at fit time.
        """
        raise NotImplementedError("Kesha, week 1: see docs/team-plan.md")
```

- [ ] **Step 4: Write the stub `src/ugr16ad/windows.py`**

```python
"""Sliding windows of consecutive minutes for the LSTM, CNN and flattened-MLP models.

A window ending at minute t holds t and the minutes just before it, never later
ones, and never spans a gap: a window that jumps the April outage would show the
model a "sequence" that never happened. Every model, including the single-minute
ones, is scored on the same ``eligible_positions`` so the comparison uses the
same minutes.
"""

from __future__ import annotations

import numpy as np


def eligible_positions(segment: np.ndarray, length: int) -> np.ndarray:
    """Row positions t where rows ``t-length+1 .. t`` all share one segment id.

    ``segment`` is ``Part.segment.to_numpy()``. Returns int64 positions in
    increasing order. With ``length == 1`` every row is eligible.
    """
    raise NotImplementedError("Kesha, week 1: see docs/team-plan.md")


def make_windows(
    X: np.ndarray, segment: np.ndarray, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stack the windows that end at each eligible position.

    Returns ``(windows, positions)``: ``windows[i]`` is ``X[positions[i]-length+1 :
    positions[i]+1]``, shape ``(m, length, features)``. When nothing is eligible,
    returns shapes ``(0, length, features)`` and ``(0,)``.
    """
    raise NotImplementedError("Kesha, week 1: see docs/team-plan.md")
```

- [ ] **Step 5: Verify the tests with a throwaway reference implementation**

Temporarily implement both files, run
`./.venv/Scripts/python.exe -m pytest tests/test_preprocess.py tests/test_windows.py --runxfail -v`
Expected: all PASS. Fix any wrong test, save the reference implementation to the scratchpad, restore the stubs exactly as in Steps 3–4.

- [ ] **Step 6: Run with stubs**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_preprocess.py tests/test_windows.py -v` → all `XFAIL`. Ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/ugr16ad/preprocess.py src/ugr16ad/windows.py tests/test_preprocess.py tests/test_windows.py
git commit -m "Add preprocessing and windowing starter files (Kesha)"
```

---

### Task 4: Vanessa — scoring (`evaluate.py`, `results.py`)

**Files:**
- Create: `src/ugr16ad/evaluate.py`
- Create: `src/ugr16ad/results.py`
- Test: `tests/test_evaluate.py`, `tests/test_results.py`

**Interfaces:**
- Consumes: `ugr16ad.paths.RESULTS_DIR`. Scores come from any scorer's `(scores, feature_errors)` (Task 5); labels from `Part.y` / `Part.labels` (Task 2).
- Produces:
  - `fit_threshold(calib_scores: np.ndarray, quantile: float = 0.99) -> float`
  - `threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]` with keys `tp, fp, fn, tn, recall, precision, f1, fpr, alert_rate`
  - `ranking_metrics(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> dict[str, float]` with keys `average_precision, recall_at_max_fpr, partial_auc`
  - `per_category_recall(labels: pd.DataFrame, alerts: np.ndarray) -> pd.DataFrame` (index = category; columns `n_positive, n_detected, recall`)
  - `SCORE_COLUMNS = ("timestamp", "part", "score")`, `REQUIRED_META = ("model", "seed", "threshold")`
  - `run_dir(model: str, seed: int, root: Path = RESULTS_DIR) -> Path`
  - `save_run(directory: Path, *, scores: pd.DataFrame, meta: dict, latents: np.ndarray | None = None, feature_errors: np.ndarray | None = None) -> None`
  - `Run` frozen dataclass (`scores`, `meta`, `latents`, `feature_errors`) and `load_run(directory: Path) -> Run`

- [ ] **Step 1: Write `tests/test_evaluate.py`**

```python
import numpy as np
import pandas as pd
from stubs import todo

from ugr16ad.evaluate import (
    fit_threshold,
    per_category_recall,
    ranking_metrics,
    threshold_metrics,
)


@todo
def test_fit_threshold_is_the_99th_percentile():
    scores = np.arange(1, 101, dtype=float)
    assert fit_threshold(scores) == np.quantile(scores, 0.99)


@todo
def test_threshold_metrics_counts_and_rates():
    y = np.array([0, 0, 0, 1, 1], dtype=bool)
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    m = threshold_metrics(y, s, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["recall"] == 0.5
    assert m["precision"] == 0.5
    assert m["f1"] == 0.5
    assert m["fpr"] == 1 / 3
    assert m["alert_rate"] == 0.4


@todo
def test_threshold_metrics_with_no_alerts():
    y = np.array([0, 1], dtype=bool)
    m = threshold_metrics(y, np.array([0.1, 0.2]), threshold=1.0)
    assert m["precision"] == 0.0
    assert m["f1"] == 0.0
    assert m["alert_rate"] == 0.0


@todo
def test_ranking_metrics_perfect_ranking():
    y = np.array([0, 0, 1, 1], dtype=bool)
    m = ranking_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m == {"average_precision": 1.0, "recall_at_max_fpr": 1.0, "partial_auc": 1.0}


@todo
def test_per_category_recall_counts_detected_minutes():
    labels = pd.DataFrame({"labeldos": [1, 0, 2, 0], "labelanomalyspam": [0, 0, 0, 0]})
    table = per_category_recall(labels, np.array([True, False, False, False]))
    assert table.loc["labeldos"].tolist() == [2, 1, 0.5]
    assert list(table.columns) == ["n_positive", "n_detected", "recall"]


@todo
def test_per_category_recall_absent_category_is_nan():
    labels = pd.DataFrame({"labelanomalyspam": [0, 0]})
    table = per_category_recall(labels, np.array([True, False]))
    assert table.loc["labelanomalyspam", "n_positive"] == 0
    assert np.isnan(table.loc["labelanomalyspam", "recall"])
```

- [ ] **Step 2: Write `tests/test_results.py`**

```python
import numpy as np
import pandas as pd
import pytest
from stubs import todo

from ugr16ad.results import load_run, run_dir, save_run

SCORES = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(["2016-07-28 00:00", "2016-07-28 00:01", "2016-07-28 00:02"]),
        "part": ["test", "test", "test"],
        "score": [0.1, 2.5, 0.3],
    }
)
META = {"model": "pca", "seed": 0, "threshold": 1.5}


@todo
def test_run_dir_layout(tmp_path):
    assert run_dir("pca", 0, root=tmp_path) == tmp_path / "pca" / "seed0"


@todo
def test_save_then_load_round_trips(tmp_path):
    latents = np.arange(6, dtype=np.float32).reshape(3, 2)
    errors = np.ones((3, 4), dtype=np.float32)
    save_run(tmp_path, scores=SCORES, meta=META, latents=latents, feature_errors=errors)
    run = load_run(tmp_path)
    pd.testing.assert_frame_equal(run.scores, SCORES, check_dtype=False)
    assert pd.api.types.is_datetime64_any_dtype(run.scores["timestamp"])
    assert run.meta == META
    np.testing.assert_array_equal(run.latents, latents)
    np.testing.assert_array_equal(run.feature_errors, errors)


@todo
def test_optional_arrays_load_as_none(tmp_path):
    save_run(tmp_path, scores=SCORES, meta=META)
    run = load_run(tmp_path)
    assert run.latents is None
    assert run.feature_errors is None


@todo
def test_save_run_rejects_missing_score_column(tmp_path):
    with pytest.raises(ValueError):
        save_run(tmp_path, scores=SCORES.drop(columns="part"), meta=META)


@todo
def test_save_run_rejects_meta_without_threshold(tmp_path):
    with pytest.raises(ValueError):
        save_run(tmp_path, scores=SCORES, meta={"model": "pca", "seed": 0})


@todo
def test_save_run_rejects_misaligned_latents(tmp_path):
    with pytest.raises(ValueError):
        save_run(tmp_path, scores=SCORES, meta=META, latents=np.zeros((2, 2)))
```

- [ ] **Step 3: Write the stub `src/ugr16ad/evaluate.py`**

```python
"""Turn anomaly scores into the numbers the report quotes.

Two kinds of result are kept apart on purpose. Metrics at the frozen threshold
are what an analyst would actually have seen. Ranking metrics sweep every
threshold on test, so they are retrospective comparisons only and never replace
the frozen threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_threshold(calib_scores: np.ndarray, quantile: float = 0.99) -> float:
    """The alarm threshold: the ``quantile`` of calibration scores (``np.quantile``).

    At 0.99 about 1% of calibration minutes alert. The threshold is then frozen
    and applied unchanged to test.
    """
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")


def threshold_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Confusion counts and rates at one threshold. A minute alerts when ``score > threshold``.

    Keys: ``tp, fp, fn, tn, recall, precision, f1, fpr, alert_rate``.
    ``alert_rate`` is the share of all minutes that alert; it is the analyst's
    workload. With no alerts, precision and F1 are 0.0 rather than undefined,
    so a silent model scores as useless instead of crashing the table.
    """
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")


def ranking_metrics(
    y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01
) -> dict[str, float]:
    """Threshold-free comparisons on test.

    Keys:
        average_precision: ``sklearn.metrics.average_precision_score``.
        recall_at_max_fpr: best recall over thresholds whose FPR <= ``max_fpr``
            (from ``sklearn.metrics.roc_curve``).
        partial_auc: ``roc_auc_score(..., max_fpr=max_fpr)``, the standardised
            (McClish) partial AUC over FPR 0..``max_fpr``.
    """
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")


def per_category_recall(labels: pd.DataFrame, alerts: np.ndarray) -> pd.DataFrame:
    """Recall per attack type, counted in minutes, not episodes.

    ``labels`` is ``Part.labels`` (one count column per category); a minute is
    positive for a category when its count is > 0. Returns one row per column of
    ``labels`` (index = category) with ``n_positive``, ``n_detected`` and
    ``recall``. ``recall`` is NaN when ``n_positive`` is 0, because SSH scan and
    spam never occur in test and "0% detected" would be a false claim.
    """
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")
```

- [ ] **Step 4: Write the stub `src/ugr16ad/results.py`**

```python
"""One file format for every model's results, so charts never depend on model code.

Each run writes to ``reports/results/<model>/seed<seed>/``:

- ``scores.csv``: one row per scored minute; columns ``timestamp, part, score``.
- ``meta.json``: at least ``model``, ``seed`` and ``threshold``, plus anything
  else worth reporting (settings, training seconds, parameter count).
- ``latents.npy`` (optional): row i is the embedding of scores row i.
- ``feature_errors.npy`` (optional): row i is the per-feature squared error of
  scores row i, for "which features caused this alert".

Charts and tables read only these files. That lets the charts be built on PCA
in week 2 and rerun unchanged on every other model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ugr16ad.paths import RESULTS_DIR

SCORE_COLUMNS = ("timestamp", "part", "score")
REQUIRED_META = ("model", "seed", "threshold")


@dataclass(frozen=True)
class Run:
    """A loaded run. ``latents`` and ``feature_errors`` are None when not saved."""

    scores: pd.DataFrame
    meta: dict
    latents: np.ndarray | None
    feature_errors: np.ndarray | None


def run_dir(model: str, seed: int, root: Path = RESULTS_DIR) -> Path:
    """``root / model / f"seed{seed}"``. Does not create it."""
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")


def save_run(
    directory: Path,
    *,
    scores: pd.DataFrame,
    meta: dict,
    latents: np.ndarray | None = None,
    feature_errors: np.ndarray | None = None,
) -> None:
    """Write a run in the format above, creating ``directory`` if needed.

    Raises:
        ValueError: if ``scores`` lacks a ``SCORE_COLUMNS`` column, ``meta``
            lacks a ``REQUIRED_META`` key, or an optional array's row count
            differs from ``len(scores)``. Bad files caught here would otherwise
            surface as a wrong chart two weeks later.
    """
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")


def load_run(directory: Path) -> Run:
    """Read a run written by ``save_run``; ``timestamp`` comes back as datetime."""
    raise NotImplementedError("Vanessa, week 1: see docs/team-plan.md")
```

- [ ] **Step 5: Verify the tests with a throwaway reference implementation**

Temporarily implement both files, run
`./.venv/Scripts/python.exe -m pytest tests/test_evaluate.py tests/test_results.py --runxfail -v`
Expected: all PASS. Fix any wrong test, save the reference implementation to the scratchpad, restore the stubs exactly as in Steps 3–4.

- [ ] **Step 6: Run with stubs**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_evaluate.py tests/test_results.py -v` → all `XFAIL`. Ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/ugr16ad/evaluate.py src/ugr16ad/results.py tests/test_evaluate.py tests/test_results.py
git commit -m "Add scoring and results-format starter files (Vanessa)"
```

---

### Task 5: Chen Lei — training tools and PCA (`train.py`, `models/pca.py`)

**Files:**
- Create: `src/ugr16ad/train.py`
- Create: `src/ugr16ad/models/__init__.py`
- Create: `src/ugr16ad/models/pca.py`
- Test: `tests/test_train.py`, `tests/test_pca.py`

**Interfaces:**
- Consumes: arrays from `Preprocessor.transform` (2-D) or `make_windows` (3-D), Task 3.
- Produces:
  - `set_seed(seed: int) -> None`
  - `TrainResult` frozen dataclass: `model: nn.Module`, `history: pd.DataFrame` (columns `epoch, train_loss, tune_loss`, epochs from 1), `best_epoch: int`, `seconds: float`
  - `train_autoencoder(model: nn.Module, train: np.ndarray, tune: np.ndarray, *, epochs: int = 50, batch_size: int = 256, lr: float = 1e-3, patience: int = 5, seed: int = 0, device: str = "cpu") -> TrainResult`
  - `score_autoencoder(model: nn.Module, x: np.ndarray, *, batch_size: int = 1024, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]`
  - `class PCAScorer(n_components: int)` with `fit(X) -> PCAScorer`, `score(X) -> tuple[np.ndarray, np.ndarray]`, `transform(X) -> np.ndarray`, attribute `explained_variance_ratio_: np.ndarray`

- [ ] **Step 1: Write `tests/test_train.py`**

```python
import numpy as np
import torch
from stubs import todo
from torch import nn

from ugr16ad.train import score_autoencoder, set_seed, train_autoencoder


class TinyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 4))

    def forward(self, x):
        return self.net(x)


class Zeros(nn.Module):
    """Reconstructs everything as 0, so the error equals the input squared."""

    def forward(self, x):
        return torch.zeros_like(x)


def _data():
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(200, 2)) @ rng.normal(size=(2, 4))).astype(np.float32)
    return x[:150], x[150:]


@todo
def test_training_lowers_loss_and_keeps_best_epoch():
    train, tune = _data()
    set_seed(0)
    result = train_autoencoder(TinyAE(), train, tune, epochs=20, batch_size=32, lr=1e-2, seed=0)
    h = result.history
    assert list(h.columns) == ["epoch", "train_loss", "tune_loss"]
    assert h["epoch"].iloc[0] == 1
    assert h["train_loss"].iloc[-1] < h["train_loss"].iloc[0]
    assert result.best_epoch == int(h.loc[h["tune_loss"].idxmin(), "epoch"])
    assert result.seconds > 0


@todo
def test_early_stopping_after_patience_epochs_without_improvement():
    train, tune = _data()
    set_seed(0)
    result = train_autoencoder(TinyAE(), train, tune, epochs=50, lr=0.0, patience=3, seed=0)
    assert len(result.history) == 4  # best at epoch 1, then 3 epochs with no improvement


@todo
def test_same_seed_same_history():
    train, tune = _data()
    runs = []
    for _ in range(2):
        set_seed(0)
        runs.append(train_autoencoder(TinyAE(), train, tune, epochs=5, seed=0).history)
    assert runs[0].equals(runs[1])


@todo
def test_score_single_minutes():
    x = np.array([[1, 2], [3, 0]], dtype=np.float32)
    scores, errors = score_autoencoder(Zeros(), x)
    np.testing.assert_allclose(errors, x**2)
    np.testing.assert_allclose(scores, [2.5, 4.5])


@todo
def test_score_windows_uses_final_minute_only():
    x = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    scores, errors = score_autoencoder(Zeros(), x)
    np.testing.assert_allclose(errors, x[:, -1, :] ** 2)
    np.testing.assert_allclose(scores, (x[:, -1, :] ** 2).mean(axis=1))
```

- [ ] **Step 2: Write `tests/test_pca.py`**

```python
import numpy as np
from stubs import todo

from ugr16ad.models.pca import PCAScorer


def _low_rank(n=100):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, 2)) @ rng.normal(size=(2, 5))


@todo
def test_pca_reconstructs_its_own_subspace():
    X = _low_rank()
    scores, errors = PCAScorer(2).fit(X).score(X)
    assert scores.shape == (100,)
    assert errors.shape == (100, 5)
    np.testing.assert_allclose(scores, 0, atol=1e-8)


@todo
def test_point_off_the_subspace_scores_higher():
    X = _low_rank()
    pca = PCAScorer(2).fit(X)
    off = X[:1] + np.array([[5.0, -5.0, 5.0, -5.0, 5.0]])
    assert pca.score(off)[0][0] > pca.score(X)[0].max()


@todo
def test_score_is_mean_of_feature_errors():
    X = _low_rank()
    noisy = X + np.random.default_rng(1).normal(scale=0.1, size=X.shape)
    scores, errors = PCAScorer(1).fit(X).score(noisy)
    np.testing.assert_allclose(scores, errors.mean(axis=1))


@todo
def test_transform_and_explained_variance():
    pca = PCAScorer(2).fit(_low_rank())
    assert pca.transform(_low_rank()).shape == (100, 2)
    assert pca.explained_variance_ratio_.shape == (2,)
    np.testing.assert_allclose(pca.explained_variance_ratio_.sum(), 1.0, atol=1e-8)
```

- [ ] **Step 3: Write the stub `src/ugr16ad/train.py`**

```python
"""One training loop and one scoring rule for every neural autoencoder.

Sharing them is what keeps the MLP, LSTM and CNN comparison fair: same loss,
same early stopping, same seeds, same "final minute" score. A model owner only
writes the ``nn.Module``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch import nn


def set_seed(seed: int) -> None:
    """Seed Python's ``random``, NumPy and PyTorch, so a recorded seed reproduces a run.

    Call it before building the model, so weight initialisation is seeded too.
    """
    raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")


@dataclass(frozen=True)
class TrainResult:
    """What a training run returns.

    Attributes:
        model: the model with its best-epoch weights restored.
        history: one row per epoch run; columns ``epoch`` (from 1),
            ``train_loss``, ``tune_loss`` (mean MSE). This is the loss curve the
            report shows.
        best_epoch: the epoch with the lowest ``tune_loss``.
        seconds: wall-clock training time, for the runtime comparison.
    """

    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    seconds: float


def train_autoencoder(
    model: nn.Module,
    train: np.ndarray,
    tune: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 0,
    device: str = "cpu",
) -> TrainResult:
    """Fit ``model`` to reconstruct its input (MSE loss, Adam optimiser).

    ``train`` and ``tune`` are 2-D ``(rows, features)`` or 3-D ``(windows,
    length, features)``. Mini-batches are shuffled within the fit part only,
    using ``seed``; that never mixes rows across parts. Stops when ``tune_loss``
    has not strictly improved for ``patience`` epochs, then restores the
    best-epoch weights.
    """
    raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")


def score_autoencoder(
    model: nn.Module, x: np.ndarray, *, batch_size: int = 1024, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, feature_errors)`` for each row or window of ``x``.

    ``feature_errors`` is the squared reconstruction error of the final minute,
    shape ``(n, features)``: ``x`` itself for 2-D input, ``x[:, -1, :]`` for
    windows. ``scores`` is its mean over features. Only the final minute counts,
    so a window model's alert refers to minute t, as a single-minute model's does.
    """
    raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")
```

- [ ] **Step 4: Write `src/ugr16ad/models/__init__.py` and the stub `src/ugr16ad/models/pca.py`**

`src/ugr16ad/models/__init__.py`:

```python
"""Anomaly scorers. Each returns ``(scores, feature_errors)`` like ``train.score_autoencoder``."""
```

`src/ugr16ad/models/pca.py`:

```python
"""The linear baseline: reconstruct each minute from its top principal components.

Every autoencoder has to beat this to justify its extra complexity. Its score
matches ``train.score_autoencoder`` exactly, so it plugs into the same
evaluation and charts.
"""

from __future__ import annotations

import numpy as np


class PCAScorer:
    """Wraps ``sklearn.decomposition.PCA``.

    Attributes:
        explained_variance_ratio_: share of fit-part variance per component.
            Used to choose ``n_components``. Set by ``fit``.
    """

    explained_variance_ratio_: np.ndarray

    def __init__(self, n_components: int) -> None:
        raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")

    def fit(self, X: np.ndarray) -> PCAScorer:
        """Fit on the fit part's preprocessed rows. Returns self."""
        raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(scores, feature_errors)``: squared error of ``inverse_transform(transform(X))``
        per feature, and its mean per row."""
        raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Component coordinates ``(rows, n_components)``, for the latent-space plots."""
        raise NotImplementedError("Chen Lei, week 1: see docs/team-plan.md")
```

- [ ] **Step 5: Verify the tests with a throwaway reference implementation**

Temporarily implement both files, run
`./.venv/Scripts/python.exe -m pytest tests/test_train.py tests/test_pca.py --runxfail -v`
Expected: all PASS. Fix any wrong test, save the reference implementation to the scratchpad, restore the stubs exactly as in Steps 3–4.

- [ ] **Step 6: Run with stubs**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_train.py tests/test_pca.py -v` → all `XFAIL`. Ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/ugr16ad/train.py src/ugr16ad/models tests/test_train.py tests/test_pca.py
git commit -m "Add training loop and PCA starter files (Chen Lei)"
```

---

### Task 6: Team plan, docs and push

**Files:**
- Create: `docs/team-plan.md`
- Modify: `README.md` (Layout block and the "No pipeline or model code exists yet" sentence)
- Modify: `CLAUDE.md` ("No pipeline or model code exists yet" paragraph and the Commands block)

**Interfaces:**
- Consumes: every name produced by Tasks 2–5.
- Produces: the ownership map members read first.

- [ ] **Step 1: Write `docs/team-plan.md`**

````markdown
# Team plan: 12 Oct – 5 Nov 2026

Basically in week 1 each of us builds one part of the common pipeline that every
model will use (data loading, input prep, scoring, training tools), then from
week 2 each of us owns one model.

## Who owns which files

| Person | Week 1: common pipeline | Tests that define "done" | Week 2–3: model and analysis |
|---|---|---|---|
| Phone | `src/ugr16ad/data.py`, `src/ugr16ad/synthetic.py` | `tests/test_data.py`, `tests/test_synthetic.py` | LSTM autoencoder (`src/ugr16ad/models/lstm.py`), GRU if time |
| Kesha | `src/ugr16ad/preprocess.py`, `src/ugr16ad/windows.py` | `tests/test_preprocess.py`, `tests/test_windows.py` | 1D-CNN autoencoder (`src/ugr16ad/models/cnn.py`) |
| Vanessa | `src/ugr16ad/evaluate.py`, `src/ugr16ad/results.py` | `tests/test_evaluate.py`, `tests/test_results.py` | Charts: score over time, loss curves, which features cause alerts, latent space; synthetic-anomaly check on every model |
| Chen Lei | `src/ugr16ad/train.py`, `src/ugr16ad/models/pca.py` | `tests/test_train.py`, `tests/test_pca.py` | PCA tuning, MLP autoencoder (`src/ugr16ad/models/mlp.py`), MLP on flattened windows |

Week 2 files don't exist yet; each owner creates theirs.

## How the pieces connect

```
data.load_dataset()        -> {"fit", "tune", "calib", "test"} Parts      (Phone)
Preprocessor.fit(fit.X)    -> transform every part                         (Kesha)
make_windows(...)          -> windows for LSTM / CNN / flattened MLP       (Kesha)
train_autoencoder(...)     -> trained model + loss curve                   (Chen Lei)
score_autoencoder / PCAScorer.score -> (scores, feature_errors)            (Chen Lei)
fit_threshold(calib scores), threshold_metrics, ranking_metrics            (Vanessa)
save_run(...)              -> reports/results/<model>/seed<seed>/          (Vanessa)
```

Every function's inputs and outputs are written in its docstring. Code against
those, not against someone's unfinished implementation, and nobody waits.

## How to do your week 1 part

1. `git checkout main && git pull`, then `git checkout -b <yourname>/week1`.
2. Set up once: `python -m venv .venv`, then `pip install -r requirements.txt` and `pip install -e .`.
3. Open your files. Replace each `raise NotImplementedError(...)` with real code.
4. Run your tests: `python -m pytest tests/test_<yourfile>.py -v`.
   - `XFAIL` = still a stub. `FAILED` with an assertion = your code disagrees with the test.
   - `FAILED ... XPASS(strict)` = it works. Delete the `@todo` line above that test.
5. Done when your tests pass with no `@todo` left, and `ruff check src tests` plus `ruff format src tests` are clean.
6. Push your branch and open a pull request into `main`. CI must be green.

If you think a test or a signature is wrong, say so in the group chat before
changing it. Someone else is coding against it.

## Open decisions (settle on the Mon 19 Oct call)

- **Tune / calibrate split of the validation rows.** `data.TUNE_FRACTION` is a 0.5 placeholder.
- **Window length.** The proposal says it is fixed before evaluation, from operational reasoning and prior literature (Phone + Kesha to propose).
- **Synthetic anomaly size.** `rate`, `magnitude`, `n_features` for the sanity check (Phone to propose).

## Schedule

| When | What |
|---|---|
| 12–18 Oct | Week 1: common pipeline (table above) |
| Mon 19 Oct | Call: merge week 1, settle open decisions |
| 19–25 Oct | Week 2: first version of each model; charts built on PCA |
| Mon 26 Oct | Call: first results; slides start |
| 26 Oct–1 Nov | Week 3: fix settings, run 3 seeds; report sections |
| Mon 2 Nov | Call: results review |
| Tue 3 Nov | Results lock |
| Wed 4 Nov | Timed rehearsal (10 min + Q&A) |
| Thu 5 Nov | Presentation and report |

Daily: one line in the group chat (done / next / blocked).
Whatever you build, write its report paragraph the same week.
````

- [ ] **Step 2: Update `README.md`**

Replace the sentence
`No pipeline or model code exists yet. The repo holds the data, the EDA script, path and logging helpers, and the tests that keep them honest.`
with
`Week 1 starter files are in place: each pipeline module has its full interface and tests, waiting to be implemented. Who owns what: **[docs/team-plan.md](docs/team-plan.md)**.`

Replace the `src/ugr16ad/` line of the Layout block with:

```
src/ugr16ad/      The package: data, preprocess, windows, train, evaluate, results, models/
```

- [ ] **Step 3: Update `CLAUDE.md`**

Replace the paragraph
```
No pipeline or model code exists yet. `src/ugr16ad/` has only `paths.py` and
`logging_utils.py`.
```
with
```
Week 1 starter files are in place: each pipeline module in `src/ugr16ad/` has its
interface, a `NotImplementedError` body naming its owner, and tests marked `@todo`
(strict xfail, from `tests/stubs.py`). Ownership and the open decisions are in
`docs/team-plan.md`. Do not implement a member's stub unless asked; the stubs are
their week-1 work.
```

- [ ] **Step 4: Full verification**

Run: `./.venv/Scripts/python.exe -m pytest -v`
Expected: the 3 path tests and `test_attack_labels_exclude_blacklist` PASS; every other test XFAIL; 0 failed.
Run: `ruff check src tests` and `ruff format --check src tests` → clean.
Run: `git grep -n "NotImplementedError" src/ugr16ad` → each line names Phone, Kesha, Vanessa or Chen Lei.

- [ ] **Step 5: Commit, push, fast-forward main**

```bash
git add docs/team-plan.md README.md CLAUDE.md docs/superpowers/plans/2026-10-11-week1-starter-files.md
git commit -m "Add team plan and point docs at the week 1 starter files"
git push origin working
git checkout main && git merge --ff-only working && git push origin main && git checkout working
```
