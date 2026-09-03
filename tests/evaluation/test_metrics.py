"""Tests for the ranking metrics.

The tie-policy tests below are the executable specification for
``order_by_score``. They fail until that function is implemented - which is the
point: they say exactly what the three policies must do.
"""

from __future__ import annotations

import numpy as np
import pytest

from vulnprio.evaluation import metrics

# Four items, three of them tied at the top score. Two are positive.
TIED_SCORES = np.array([9.8, 9.8, 9.8, 5.0])
TIED_LABELS = np.array([0, 1, 0, 1])


class TestKFromPercent:
    def test_rounds_up(self):
        assert metrics.k_from_percent(1000, 10) == 100
        assert metrics.k_from_percent(1005, 10) == 101

    def test_at_least_one(self):
        assert metrics.k_from_percent(10, 0.5) == 1

    @pytest.mark.parametrize("bad", [0, -5, 101])
    def test_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError):
            metrics.k_from_percent(100, bad)


class TestTiePolicies:
    def test_pessimistic_puts_negatives_first_within_ties(self):
        order = metrics.order_by_score(TIED_SCORES, TIED_LABELS, tie_policy="pessimistic")
        # Top-2 are drawn from the tied 9.8 group; pessimistic means both negatives.
        assert TIED_LABELS[order[0]] == 0
        assert TIED_LABELS[order[1]] == 0

    def test_optimistic_puts_positives_first_within_ties(self):
        order = metrics.order_by_score(TIED_SCORES, TIED_LABELS, tie_policy="optimistic")
        assert order[0] == 1
        assert TIED_LABELS[order[0]] == 1

    def test_policies_never_reorder_across_score_groups(self):
        # The 5.0 item is last under every policy: ties are broken only *within*
        # a score group, never across groups.
        for policy in ("pessimistic", "optimistic", "random"):
            order = metrics.order_by_score(TIED_SCORES, TIED_LABELS, tie_policy=policy)
            assert order[-1] == 3, f"{policy} moved a strictly-lower score out of last place"

    def test_random_is_reproducible_under_a_seed(self):
        a = metrics.order_by_score(TIED_SCORES, TIED_LABELS, tie_policy="random", seed=1)
        b = metrics.order_by_score(TIED_SCORES, TIED_LABELS, tie_policy="random", seed=1)
        np.testing.assert_array_equal(a, b)

    def test_precision_at_k_differs_by_policy(self):
        # The motivating case: identical scores, different reported precision.
        pess = metrics.precision_at_k(TIED_LABELS, TIED_SCORES, 50, tie_policy="pessimistic")
        opt = metrics.precision_at_k(TIED_LABELS, TIED_SCORES, 50, tie_policy="optimistic")
        assert pess < opt


class TestPrecisionAndRecall:
    def test_perfect_ranking_gives_full_recall_at_small_k(self, imbalanced_labels):
        y, _ = imbalanced_labels
        perfect = y.astype(float)
        assert metrics.recall_at_k(y, perfect, 1, tie_policy="optimistic") == 1.0

    def test_random_precision_approximates_base_rate(self):
        rng = np.random.default_rng(0)
        n = 20_000
        y = (rng.random(n) < 0.01).astype(int)
        scores = rng.random(n)
        p = metrics.precision_at_k(y, scores, 10)
        assert abs(p - y.mean()) < 0.005

    def test_lift_of_one_means_no_better_than_random(self):
        y = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        lift = metrics.lift_at_k(y, scores, 50)
        assert lift == pytest.approx(1.0)

    def test_recall_is_nan_without_positives(self):
        y = np.zeros(10, dtype=int)
        assert np.isnan(metrics.recall_at_k(y, np.arange(10, dtype=float), 10))


class TestEvaluateRanking:
    def test_one_row_per_budget_with_expected_columns(self, imbalanced_labels):
        y, scores = imbalanced_labels
        table = metrics.evaluate_ranking(y, scores, [1, 5, 10])
        assert len(table) == 3
        for col in ("precision_at_k", "recall_at_k", "lift_at_k", "pr_auc", "base_rate"):
            assert col in table.columns

    def test_recall_is_monotone_in_budget(self, imbalanced_labels):
        y, scores = imbalanced_labels
        table = metrics.evaluate_ranking(y, scores, [1, 5, 10, 20, 50])
        assert table["recall_at_k"].is_monotonic_increasing


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self, imbalanced_labels):
        y, scores = imbalanced_labels
        point, lo, hi = metrics.bootstrap_ci(y, scores, 10, n_iterations=200)
        assert lo <= point <= hi
