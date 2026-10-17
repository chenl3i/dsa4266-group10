import numpy as np
import pandas as pd
import pytest

from ugr16ad.evaluate import (
    fit_threshold,
    per_category_recall,
    ranking_metrics,
    threshold_metrics,
)


def test_fit_threshold_is_the_99th_percentile():
    scores = np.arange(1, 101, dtype=float)
    assert fit_threshold(scores) == np.quantile(scores, 0.99)


def test_fit_threshold_honours_the_quantile_argument():
    scores = np.arange(1, 101, dtype=float)
    assert fit_threshold(scores, quantile=0.5) == np.quantile(scores, 0.5)
    assert fit_threshold(scores, quantile=0.5) != fit_threshold(scores)


def test_fit_threshold_returns_a_plain_float_for_float32_scores():
    # np.quantile of float32 scores is np.float32, which json.dump rejects in meta.json.
    scores = np.arange(1, 101, dtype=np.float32)
    assert type(fit_threshold(scores)) is float


def test_threshold_metrics_counts_and_rates():
    y = np.array([0, 0, 0, 1, 1], dtype=bool)
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    m = threshold_metrics(y, s, threshold=0.5)
    assert set(m) == {"tp", "fp", "fn", "tn", "recall", "precision", "f1", "fpr", "alert_rate"}
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["recall"] == 0.5
    assert m["precision"] == 0.5
    assert m["f1"] == 0.5
    assert m["fpr"] == 1 / 3
    assert m["alert_rate"] == 0.4


def test_a_score_equal_to_the_threshold_does_not_alert():
    # The threshold is a calibration quantile, so some calibration minute sits exactly on
    # it. "score > threshold" keeps that minute quiet; ">=" would alert on it.
    y = np.array([0, 1], dtype=bool)
    m = threshold_metrics(y, np.array([0.5, 0.5]), threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (0, 0, 1, 1)
    assert m["alert_rate"] == 0.0


def test_threshold_metrics_with_no_alerts():
    y = np.array([0, 1], dtype=bool)
    m = threshold_metrics(y, np.array([0.1, 0.2]), threshold=1.0)
    assert m["precision"] == 0.0
    assert m["f1"] == 0.0
    assert m["alert_rate"] == 0.0


def test_threshold_metrics_f1_is_the_harmonic_mean_when_recall_and_precision_differ():
    # tp=2, fp=0, fn=2 -> precision 1.0, recall 0.5, f1 2/3 (an arithmetic mean would be 0.75).
    y = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    s = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1])
    m = threshold_metrics(y, s, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert m["f1"] == pytest.approx(2 / 3)


def test_ranking_metrics_perfect_ranking():
    y = np.array([0, 0, 1, 1], dtype=bool)
    m = ranking_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m == {"average_precision": 1.0, "recall_at_max_fpr": 1.0, "partial_auc": 1.0}


# Ranked high to low the minutes read: positive, negative, positive, then three negatives.
# The ROC curve passes (fpr 0, tpr 0.5), (0.25, 0.5), (0.25, 1.0), so every number below is
# worked out by hand rather than copied from sklearn.
RANKED_Y = np.array([0, 0, 0, 0, 1, 1], dtype=bool)
RANKED_S = np.array([0.1, 0.2, 0.3, 0.6, 0.5, 0.9])


def test_ranking_metrics_imperfect_ranking():
    m = ranking_metrics(RANKED_Y, RANKED_S, max_fpr=0.25)
    assert m["average_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert m["recall_at_max_fpr"] == 1.0
    # McClish: 0.5 * (1 + (0.125 - 0.25**2 / 2) / (0.25 - 0.25**2 / 2)).
    assert m["partial_auc"] == pytest.approx(0.5 * (1 + 0.09375 / 0.21875))


def test_recall_at_max_fpr_only_counts_thresholds_within_the_budget():
    # fpr stays 0 only until the first negative outranks a positive: just half the
    # positives are found inside a budget of 0.2, and fpr == 0.25 is within a budget of 0.25.
    assert ranking_metrics(RANKED_Y, RANKED_S, max_fpr=0.2)["recall_at_max_fpr"] == 0.5
    assert ranking_metrics(RANKED_Y, RANKED_S, max_fpr=0.25)["recall_at_max_fpr"] == 1.0


def test_ranking_metrics_inverted_ranking_finds_nothing_inside_the_budget():
    m = ranking_metrics(RANKED_Y, -RANKED_S, max_fpr=0.25)
    assert m["recall_at_max_fpr"] == 0.0
    assert m["average_precision"] < 0.5


def test_per_category_recall_counts_detected_minutes():
    labels = pd.DataFrame({"labeldos": [1, 0, 2, 0], "labelanomalyspam": [0, 0, 0, 0]})
    table = per_category_recall(labels, np.array([True, False, False, False]))
    assert table.loc["labeldos"].tolist() == [2, 1, 0.5]
    assert list(table.columns) == ["n_positive", "n_detected", "recall"]


def test_per_category_recall_scores_each_category_against_its_own_minutes():
    # Alerts on minutes that are negative for a category must not count as detections of it.
    labels = pd.DataFrame({"labeldos": [1, 0, 2, 0], "label_scan": [0, 1, 0, 1]})
    table = per_category_recall(labels, np.array([True, True, True, False]))
    assert table.loc["labeldos"].tolist() == [2, 2, 1.0]
    assert table.loc["label_scan"].tolist() == [2, 1, 0.5]


def test_per_category_recall_index_follows_label_column_order():
    labels = pd.DataFrame({"labeldos": [1, 0], "labelnerisbotnet": [0, 1], "labelscan11": [1, 1]})
    table = per_category_recall(labels, np.array([True, False]))
    assert list(table.index) == ["labeldos", "labelnerisbotnet", "labelscan11"]


def test_per_category_recall_absent_category_is_nan():
    labels = pd.DataFrame({"labelanomalyspam": [0, 0]})
    table = per_category_recall(labels, np.array([True, False]))
    assert table.loc["labelanomalyspam", "n_positive"] == 0
    assert table.loc["labelanomalyspam", "n_detected"] == 0
    assert np.isnan(table.loc["labelanomalyspam", "recall"])


def test_nan_scores_are_rejected():
    with pytest.raises(ValueError):
        fit_threshold(np.array([1.0, np.nan]))
    y = np.array([0, 1], dtype=bool)
    with pytest.raises(ValueError):
        threshold_metrics(y, np.array([0.1, np.nan]), threshold=0.5)
    with pytest.raises(ValueError):
        ranking_metrics(y, np.array([0.1, np.nan]))


def test_per_category_recall_rejects_misaligned_alerts():
    labels = pd.DataFrame({"labeldos": [1, 0, 2, 0]})
    with pytest.raises(ValueError):
        per_category_recall(labels, np.array([True, False]))
