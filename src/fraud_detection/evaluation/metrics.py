"""Imbalance-appropriate evaluation metrics (DOC-01 §10, DOC-02 §11).

Threshold-independent metrics (PR-AUC as Average Precision, ROC-AUC) are computed on
probabilities; threshold-dependent metrics (Precision, Recall, F1, F2, confusion matrix)
at an explicit threshold using the decision rule ``p >= threshold``. Accuracy is
deliberately not computed: it is not a selection metric (D-03).

The decision rule itself is :func:`fraud_detection.inference.predictor.apply_threshold`, the
single implementation shared with serving (DOC-04 §6); it is re-exported here for existing callers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_detection.inference.predictor import apply_threshold

__all__ = ["apply_threshold", "compute_metrics", "threshold_independent_metrics", "threshold_metrics"]

POSITIVE_LABEL = 1


def _as_arrays(y_true: Any, y_proba: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(y_proba, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"y_true and y_proba must be 1-D and equal length, got {y.shape} and {p.shape}")
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("y_true must contain both classes 0 and 1")
    return y, p


def threshold_independent_metrics(y_true: Any, y_proba: Any) -> dict[str, float]:
    """PR-AUC (Average Precision, primary), ROC-AUC, prevalence and PR-AUC lift over no-skill."""
    y, p = _as_arrays(y_true, y_proba)
    pr_auc = float(average_precision_score(y, p, pos_label=POSITIVE_LABEL))
    prevalence = float(y.mean())
    return {
        "pr_auc": pr_auc,
        "roc_auc": float(roc_auc_score(y, p)),
        "prevalence": prevalence,
        "pr_auc_lift": pr_auc / prevalence,
    }


def threshold_metrics(y_true: Any, y_proba: Any, threshold: float) -> dict[str, Any]:
    """Precision, Recall, F1, F2 and confusion-matrix counts at ``threshold``."""
    y, p = _as_arrays(y_true, y_proba)
    y_pred = apply_threshold(p, threshold)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(fbeta_score(y, y_pred, beta=1.0, zero_division=0)),
        "f2": float(fbeta_score(y, y_pred, beta=2.0, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        # Rows: actual [legitimate, fraud]; columns: predicted [legitimate, fraud].
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def compute_metrics(y_true: Any, y_proba: Any, threshold: float) -> dict[str, Any]:
    """All required metrics for one dataset: threshold-independent plus at ``threshold``."""
    return {
        **threshold_independent_metrics(y_true, y_proba),
        "n_rows": int(len(y_true)),
        "n_fraud": int(np.sum(np.asarray(y_true) == POSITIVE_LABEL)),
        "at_threshold": threshold_metrics(y_true, y_proba, threshold),
    }
