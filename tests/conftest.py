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


@pytest.fixture(scope="session")
def real_parts():
    """The real dataset, loaded once for the slow pipeline checks."""
    from ugr16ad.data import load_dataset

    return load_dataset()
