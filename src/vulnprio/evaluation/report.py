"""Assemble the comparison table that answers the research question.

The deliverable is one table: rankers down the rows, patch budgets across the
columns, precision@k in the cells, with the CVSS baseline's row as the reference
line and a delta column against it.

Implemented in plan Task 16.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vulnprio.evaluation.results import RankerResult


def comparison_table(
    results: dict[str, RankerResult],
    *,
    baseline: str = "cvss",
    metric: str = "precision_at_k",
) -> pd.DataFrame:
    """Pivot per-ranker metric tables into one comparison, with deltas vs baseline.

    Args:
        results: Ranker name -> its :class:`RankerResult`.
        baseline: Which ranker is the reference line.
        metric: Which column of each ranker's table to pivot on.
    """
    raise NotImplementedError("Implemented in plan Task 16 (reporting).")


def write_results(
    results: dict[str, RankerResult], output_dir: Path, *, experiment_name: str
) -> list[Path]:
    """Write per-ranker CSVs plus the comparison table and a markdown summary."""
    raise NotImplementedError("Implemented in plan Task 16 (reporting).")
