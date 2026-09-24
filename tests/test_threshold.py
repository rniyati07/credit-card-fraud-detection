"""Metric correctness on hand-computed toy arrays (DOC-05 §15, M4 part of test_threshold.py)."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_detection.evaluation.metrics import (
    apply_threshold,
    compute_metrics,
    threshold_independent_metrics,
    threshold_metrics,
)

# Ranking by probability: 0.9 (fraud), 0.6 (legit), 0.4 (fraud), 0.1 (legit).
Y = np.array([0, 0, 1, 1])
P = np.array([0.1, 0.6, 0.4, 0.9])


def test_threshold_independent_metrics_toy() -> None:
    m = threshold_independent_metrics(Y, P)

    # AP = 0.5 * (1/1) + 0.5 * (2/3); ROC-AUC = 3 of 4 (pos, neg) pairs ranked correctly.
    assert m["pr_auc"] == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["prevalence"] == pytest.approx(0.5)
    assert m["pr_auc_lift"] == pytest.approx(m["pr_auc"] / 0.5)


def test_threshold_metrics_toy() -> None:
    m = threshold_metrics(Y, P, 0.5)  # predictions [0, 1, 0, 1]

    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["f2"] == pytest.approx(0.5)


def test_f2_weights_recall() -> None:
    # Predictions [1, 1, 1, 1]: precision 0.5, recall 1.0 -> F2 = 5*0.5*1 / (4*0.5 + 1) = 0.8333
    m = threshold_metrics(Y, P, 0.05)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["f2"] == pytest.approx(5 * 0.5 / 3)


def test_decision_rule_is_greater_or_equal() -> None:
    assert apply_threshold([0.49, 0.5, 0.51], 0.5).tolist() == [0, 1, 1]
    assert threshold_metrics(Y, P, 0.4)["tp"] == 2  # p == threshold counts as fraud


def test_perfect_ranking() -> None:
    m = threshold_independent_metrics([0, 0, 1], [0.1, 0.2, 0.9])
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_no_positive_predictions_uses_zero_division() -> None:
    m = threshold_metrics(Y, P, 0.95)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["fn"] == 2 and m["tp"] == 0


def test_compute_metrics_structure() -> None:
    m = compute_metrics(Y, P, 0.5)
    assert m["n_rows"] == 4 and m["n_fraud"] == 2
    assert set(m) >= {"pr_auc", "roc_auc", "prevalence", "pr_auc_lift", "at_threshold"}
    assert m["at_threshold"]["threshold"] == 0.5


@pytest.mark.parametrize(
    ("y", "p", "message"),
    [
        ([0, 1, 1], [0.1, 0.2], "equal length"),
        ([0, 1], [0.1, 1.2], r"\[0, 1\]"),
        ([1, 1], [0.1, 0.2], "both classes"),
    ],
)
def test_invalid_inputs_raise(y, p, message) -> None:
    with pytest.raises(ValueError, match=message):
        compute_metrics(y, p, 0.5)
