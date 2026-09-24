"""Metrics, threshold search and selection tie-break tests (DOC-05 §15: test_threshold.py, M4/M6)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import ThresholdConfig
from fraud_detection.evaluation.metrics import (
    apply_threshold,
    compute_metrics,
    threshold_independent_metrics,
    threshold_metrics,
)
from fraud_detection.evaluation.threshold import (
    OBJECTIVE_FALLBACK,
    OBJECTIVE_PRIMARY,
    analyse_thresholds,
    candidate_thresholds,
    select_threshold,
    threshold_grid,
    threshold_table,
)
from fraud_detection.models.select import CandidateSummary, select_model

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


# ================================================================ threshold search (M6)

CFG = ThresholdConfig(
    r_min=0.8, r_min_confirmed=False, grid_start=0.01, grid_stop=0.99, grid_step=0.01,
    include_pr_curve_points=True, fallback="f2", reference_threshold=0.5,
)
# 5 frauds and 5 legitimate transactions with a known ranking.
Y_CURVE = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
P_CURVE = np.array([0.95, 0.90, 0.70, 0.40, 0.20, 0.85, 0.50, 0.30, 0.10, 0.05])


def test_threshold_grid_is_exact() -> None:
    grid = threshold_grid(0.01, 0.99, 0.01)
    assert len(grid) == 99 and grid[0] == 0.01 and grid[-1] == 0.99
    assert 0.07 in grid and 0.29 in grid  # no float drift such as 0.07000000000000001


def test_candidates_include_grid_and_pr_curve_points() -> None:
    cands = candidate_thresholds(Y_CURVE, P_CURVE, CFG)
    thresholds = set(cands["threshold"])
    assert {0.01, 0.5, 0.99} <= thresholds
    assert {0.95, 0.4, 0.2} <= thresholds  # PR-curve points: the scores themselves
    assert cands.loc[cands["threshold"] == 0.5, "source"].item() == "grid+pr_curve"
    assert cands.loc[cands["threshold"] == 0.37, "source"].item() == "grid"
    assert cands["threshold"].is_monotonic_increasing
    off_grid = candidate_thresholds(np.array([0, 1, 1]), np.array([0.1, 0.123, 0.9]), CFG)
    assert off_grid.loc[off_grid["threshold"] == 0.123, "source"].item() == "pr_curve"
    no_pr = candidate_thresholds(Y_CURVE, P_CURVE, dataclasses.replace(CFG, include_pr_curve_points=False))
    assert len(no_pr) == 99 and set(no_pr["source"]) == {"grid"}


def test_candidates_exclude_thresholds_outside_open_interval() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.3, 0.6])
    cands = candidate_thresholds(y, p, CFG)
    assert cands["threshold"].between(0, 1, inclusive="neither").all()


def test_threshold_table_matches_sklearn_metrics() -> None:
    table = threshold_table(Y_CURVE, P_CURVE, candidate_thresholds(Y_CURVE, P_CURVE, CFG))
    for t in (0.2, 0.33, 0.4, 0.5, 0.85, 0.95):
        row = table[np.isclose(table["threshold"], t)]
        assert len(row) == 1
        expected = threshold_metrics(Y_CURVE, P_CURVE, row["threshold"].item())
        for key in ("tp", "fp", "tn", "fn"):
            assert row[key].item() == expected[key]
        for key in ("precision", "recall", "f1", "f2"):
            assert row[key].item() == pytest.approx(expected[key])


def test_known_curve_primary_objective() -> None:
    # Recall >= 0.8 needs 4 of 5 frauds: p >= 0.40 gives TP 4, FP 2 (0.85, 0.50) -> precision 4/6.
    _, choice = analyse_thresholds(Y_CURVE, P_CURVE, CFG)
    assert choice.objective == OBJECTIVE_PRIMARY and choice.reached_r_min
    assert choice.threshold == pytest.approx(0.40)
    assert (choice.metrics["tp"], choice.metrics["fp"]) == (4, 2)
    assert choice.metrics["precision"] == pytest.approx(4 / 6)


def test_tie_goes_to_higher_threshold() -> None:
    # Every threshold in (0.30, 0.40] gives the same counts as 0.40 on this curve.
    table = threshold_table(Y_CURVE, P_CURVE, candidate_thresholds(Y_CURVE, P_CURVE, CFG))
    tied = table[(table["tp"] == 4) & (table["fp"] == 2)]
    assert len(tied) > 1
    assert select_threshold(table, 0.8, 0.01, 0.99).threshold == tied["threshold"].max()


def test_unattainable_r_min_uses_f2_fallback() -> None:
    table = pd.DataFrame(
        {"threshold": [0.2, 0.5, 0.8], "source": "grid", "predicted_positive": [6, 3, 1],
         "tp": [3, 2, 1], "fp": [3, 1, 0], "tn": [0, 2, 3], "fn": [1, 2, 3],
         "precision": [0.5, 2 / 3, 1.0], "recall": [0.75, 0.5, 0.25],
         "f1": [0.6, 0.571, 0.4], "f2": [0.68, 0.53, 0.29]}
    )
    choice = select_threshold(table, 0.9, 0.01, 0.99)
    assert choice.objective == OBJECTIVE_FALLBACK and not choice.reached_r_min
    assert choice.threshold == 0.2


def test_fallback_tie_goes_to_higher_threshold() -> None:
    table = pd.DataFrame(
        {"threshold": [0.3, 0.6], "source": "grid", "predicted_positive": [2, 2], "tp": [1, 1],
         "fp": [1, 1], "tn": [1, 1], "fn": [1, 1], "precision": [0.5, 0.5], "recall": [0.5, 0.5],
         "f1": [0.5, 0.5], "f2": [0.5, 0.5]}
    )
    assert select_threshold(table, 0.9, 0.01, 0.99).threshold == 0.6


def test_grid_edge_is_reported_not_forced() -> None:
    y = np.array([1, 0, 0, 0])
    p = np.array([0.999, 0.1, 0.2, 0.3])
    grid_only = dataclasses.replace(CFG, include_pr_curve_points=False)
    choice = select_threshold(threshold_table(y, p, candidate_thresholds(y, p, grid_only)), 0.8, 0.01, 0.99)
    assert choice.threshold == 0.99 and choice.at_grid_edge


# ======================================================================= selection (M6)


def _cand(name, pr_auc, fp=5, size=100, reached=True) -> CandidateSummary:
    return CandidateSummary(
        name=name, val_pr_auc=pr_auc, reached_r_min=reached,
        objective=OBJECTIVE_PRIMARY if reached else OBJECTIVE_FALLBACK,
        threshold=0.5, precision=0.9, recall=0.85, f2=0.8, fp=fp, artifact_bytes=size,
    )


def test_selection_by_pr_auc_when_no_tie() -> None:
    result = select_model({"logistic_regression": _cand("logistic_regression", 0.66),
                           "random_forest": _cand("random_forest", 0.80, fp=1),
                           "xgboost": _cand("xgboost", 0.83, fp=9)}, 0.01)
    assert result.selected == "xgboost" and not result.tie_break_applied
    assert result.ranking == ["xgboost", "random_forest", "logistic_regression"]


def test_tie_break_prefers_fewer_false_positives() -> None:
    result = select_model({"logistic_regression": _cand("logistic_regression", 0.60),
                           "random_forest": _cand("random_forest", 0.825, fp=2),
                           "xgboost": _cand("xgboost", 0.830, fp=6)}, 0.01)
    assert result.pr_auc_leader == "xgboost"
    assert result.tie_group == ["xgboost", "random_forest"]
    assert result.selected == "random_forest"


def test_tie_break_then_simpler_then_smaller_artifact() -> None:
    simpler = select_model({"logistic_regression": _cand("logistic_regression", 0.80, fp=3),
                            "xgboost": _cand("xgboost", 0.805, fp=3)}, 0.01)
    assert simpler.selected == "logistic_regression"
    smaller = select_model({"logistic_regression": _cand("logistic_regression", 0.5),
                            "random_forest": _cand("random_forest", 0.80, fp=3, size=900),
                            "xgboost": _cand("xgboost", 0.805, fp=3, size=100)}, 0.01)
    assert smaller.selected == "xgboost"


def test_ineligible_models_are_excluded() -> None:
    result = select_model({"logistic_regression": _cand("logistic_regression", 0.60),
                           "xgboost": _cand("xgboost", 0.90, reached=False)}, 0.01)
    assert result.eligible == ["logistic_regression"] and result.selected == "logistic_regression"


def test_no_model_reaches_r_min_keeps_all_models() -> None:
    result = select_model({"logistic_regression": _cand("logistic_regression", 0.60, reached=False),
                           "xgboost": _cand("xgboost", 0.90, reached=False)}, 0.01)
    assert result.eligible == ["logistic_regression", "xgboost"] and result.selected == "xgboost"


def test_baseline_sanity_falls_back_to_logistic_regression() -> None:
    # A tie-break can pick a model whose PR-AUC does not exceed the LR baseline.
    result = select_model({"logistic_regression": _cand("logistic_regression", 0.800, fp=5),
                           "random_forest": _cand("random_forest", 0.795, fp=1)}, 0.01)
    assert result.sanity_passed is False and result.baseline_fallback
    assert result.selected == "logistic_regression"


def test_selection_steps_are_recorded() -> None:
    result = select_model({"logistic_regression": _cand("logistic_regression", 0.6),
                           "xgboost": _cand("xgboost", 0.8)}, 0.01)
    assert [s["step"][:2] for s in result.steps] == ["1.", "2.", "3.", "4.", "5."]
    with pytest.raises(ValueError):
        select_model({}, 0.01)
