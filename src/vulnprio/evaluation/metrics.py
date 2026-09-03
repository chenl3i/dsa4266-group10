"""Ranking metrics for the patch-budget decision problem.

The brief gets this right and it is worth restating in code: accuracy is
meaningless at a 0.5% positive rate, and ROC-AUC is misleadingly optimistic
under heavy imbalance because the false-positive rate barely moves when the
negative class is enormous. The metrics here answer the question actually being
asked - "we can patch k% of our backlog; how many real exploits does that catch?"

  precision@k  Of the k% we patched, what fraction were genuinely exploited?
               Directly interpretable as "was our effort well spent?"
  recall@k     Of all exploited vulnerabilities, what fraction did we catch?
               Directly interpretable as "how much risk did we retire?"
  lift@k       precision@k divided by the base rate. Answers "how much better
               than picking at random?" and is comparable across datasets with
               different positive rates.
  PR-AUC       Threshold-free summary (sklearn's average_precision_score).
               Preferred over ROC-AUC for the reason above.

All functions take ``y_true`` and ``scores`` as plain arrays so they can be used
on baselines and models alike, and unit-tested without any pipeline.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TiePolicy = Literal["pessimistic", "optimistic", "random"]


def k_from_percent(n: int, k_percent: float) -> int:
    """Number of items in the top ``k_percent`` of ``n``, at least 1."""
    if not 0 < k_percent <= 100:
        raise ValueError(f"k_percent must lie in (0, 100], got {k_percent}")
    return max(1, int(np.ceil(n * k_percent / 100.0)))


def order_by_score(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    tie_policy: TiePolicy = "pessimistic",
    seed: int = 4266,
) -> np.ndarray:
    """Return indices ordering items from highest to lowest score.

    Tie handling is the single most consequential choice in this module, and it
    is not a detail. CVSS base scores are a discrete formula: tens of thousands
    of CVEs share a score of exactly 9.8, and 9.8 sits above the cut for any
    realistic patch budget. So "the top 10% by CVSS" is genuinely ambiguous, and
    the reported precision@10% for the baseline can differ by a factor of two
    depending on which tied items land inside the cut.

    The three defensible policies:

    ``pessimistic``
        Within a tied group, place negatives before positives. This reports the
        *worst case* for the ranker being scored. Conservative, and the right
        default for a baseline you are trying to beat - it cannot be accused of
        flattering your own model by comparison.

    ``optimistic``
        Within a tied group, place positives first. Reports the best case.
        Useful as an upper bound; reporting it *alongside* the pessimistic
        figure honestly brackets how much of a result is real signal versus
        sort-order luck.

    ``random``
        Shuffle within tied groups using ``seed``. Reports the expected case,
        which is what a team drawing arbitrarily from a pile of 9.8s would
        actually experience. Most realistic, but adds variance - pair it with
        the bootstrap CI.

    Args:
        scores: One score per item, higher = ranked sooner.
        y_true: Binary labels, needed only to resolve ties.
        tie_policy: One of the three above.
        seed: Used only when ``tie_policy == "random"``.

    Returns:
        Integer index array of length ``len(scores)``.
    """
    # TODO(group): implement the three tie policies.
    #
    # Hint: np.lexsort sorts by the LAST key first, so a single call can express
    # all three - build a secondary key from y_true (or from a seeded random
    # draw) and lexsort on (secondary, -scores).
    #
    # Sanity check to write a test against: given
    #     scores = [9.8, 9.8, 9.8, 5.0] and y_true = [0, 1, 0, 1]
    # the top-1 under "pessimistic" must be a negative, and under "optimistic"
    # must be the positive at index 1.
    raise NotImplementedError(
        "Tie-breaking policy not yet implemented - see docs/decisions/0004-tie-breaking.md"
    )


def precision_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_percent: float,
    *,
    tie_policy: TiePolicy = "pessimistic",
    seed: int = 4266,
) -> float:
    """Fraction of the top ``k_percent`` that are truly exploited."""
    y_true = np.asarray(y_true)
    order = order_by_score(np.asarray(scores), y_true, tie_policy=tie_policy, seed=seed)
    k = k_from_percent(len(y_true), k_percent)
    return float(y_true[order[:k]].mean())


def recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_percent: float,
    *,
    tie_policy: TiePolicy = "pessimistic",
    seed: int = 4266,
) -> float:
    """Fraction of all exploited vulnerabilities caught within the top ``k_percent``."""
    y_true = np.asarray(y_true)
    total_positive = int(y_true.sum())
    if total_positive == 0:
        return float("nan")
    order = order_by_score(np.asarray(scores), y_true, tie_policy=tie_policy, seed=seed)
    k = k_from_percent(len(y_true), k_percent)
    return float(y_true[order[:k]].sum() / total_positive)


def lift_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_percent: float,
    *,
    tie_policy: TiePolicy = "pessimistic",
    seed: int = 4266,
) -> float:
    """precision@k divided by the base rate. 1.0 means no better than random."""
    y_true = np.asarray(y_true)
    base_rate = float(y_true.mean())
    if base_rate == 0:
        return float("nan")
    return precision_at_k(y_true, scores, k_percent, tie_policy=tie_policy, seed=seed) / base_rate


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision. The threshold-free summary to lead with."""
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Reported only for comparison with prior work.

    Include it in the results table so readers can situate the work, but do not
    lead with it - under 0.5% positives it looks strong almost regardless of how
    useful the ranking is, which is precisely the trap the brief flags.
    """
    return float(roc_auc_score(y_true, scores))


def evaluate_ranking(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_percents: list[float],
    *,
    tie_policy: TiePolicy = "pessimistic",
    seed: int = 4266,
) -> pd.DataFrame:
    """Full metric table for one ranker: one row per patch budget.

    Threshold-free metrics (PR-AUC, ROC-AUC) and the base rate repeat on every
    row so the table stays self-contained when a single row is quoted.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    base_rate = float(y_true.mean())
    rows = []
    for k in k_percents:
        rows.append(
            {
                "k_percent": k,
                "n_patched": k_from_percent(len(y_true), k),
                "precision_at_k": precision_at_k(
                    y_true, scores, k, tie_policy=tie_policy, seed=seed
                ),
                "recall_at_k": recall_at_k(y_true, scores, k, tie_policy=tie_policy, seed=seed),
                "lift_at_k": lift_at_k(y_true, scores, k, tie_policy=tie_policy, seed=seed),
                "base_rate": base_rate,
                "pr_auc": pr_auc(y_true, scores),
                "roc_auc": roc_auc(y_true, scores),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_percent: float,
    *,
    metric: str = "precision_at_k",
    n_iterations: int = 1000,
    confidence: float = 0.95,
    tie_policy: TiePolicy = "pessimistic",
    seed: int = 4266,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a top-k metric.

    Needed because with roughly 1,300 positives spread across a test window, the
    top-k slice can contain only a few dozen of them. A precision@10% that beats
    the baseline by two points is not a result if the intervals overlap - and
    "our model beat CVSS" is the project's headline claim, so it needs an
    interval attached.

    Returns:
        ``(point_estimate, lower, upper)``.
    """
    fn = {"precision_at_k": precision_at_k, "recall_at_k": recall_at_k, "lift_at_k": lift_at_k}[
        metric
    ]
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    point = fn(y_true, scores, k_percent, tie_policy=tie_policy, seed=seed)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        samples[i] = fn(y_true[idx], scores[idx], k_percent, tie_policy=tie_policy, seed=seed)

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return point, float(lower), float(upper)
