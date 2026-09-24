"""Validation-only threshold analysis and selection (DOC-02 §13; DOC-05 M6).

Procedure:

1. Candidate thresholds: the configured grid (0.01-0.99, step 0.01) plus, optionally,
   the unique thresholds of ``precision_recall_curve`` on the validation probabilities.
   Only thresholds strictly inside (0, 1) are kept, because a frozen threshold must
   satisfy ``0 < threshold < 1`` (DOC-04 §4, check 4).
2. For every threshold: TP, FP, TN, FN, Precision, Recall, F1, F2 with the project
   decision rule ``fraud iff p >= threshold``.
3. Objective:
   * primary (``recall>=r_min_max_precision``): among thresholds with Recall >= r_min,
     maximise Precision; ties go to the higher threshold;
   * fallback (``max_f2``): if no threshold reaches r_min, maximise F2; ties go to the
     higher threshold.

The chosen threshold is reported as-is, even at a grid edge (DOC-05 M6 "do not force").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraud_detection.config import ThresholdConfig

logger = logging.getLogger(__name__)

OBJECTIVE_PRIMARY = "recall>=r_min_max_precision"
OBJECTIVE_FALLBACK = "max_f2"
SOURCE_GRID = "grid"
SOURCE_PR_CURVE = "pr_curve"
SOURCE_BOTH = "grid+pr_curve"
TABLE_COLUMNS = (
    "threshold", "source", "predicted_positive", "tp", "fp", "tn", "fn",
    "precision", "recall", "f1", "f2",
)


@dataclass(frozen=True)
class ThresholdChoice:
    """The selected operating point of one model."""

    threshold: float
    objective: str
    r_min: float
    reached_r_min: bool
    at_grid_edge: bool
    metrics: dict[str, Any]


def threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Evenly spaced thresholds from ``start`` to ``stop`` inclusive, rounded to 10 decimals."""
    n = int(round((stop - start) / step)) + 1
    grid = np.round(start + step * np.arange(n), 10)
    return grid[grid <= stop + 1e-12]


def candidate_thresholds(y_true: Any, y_proba: Any, config: ThresholdConfig) -> pd.DataFrame:
    """Sorted unique candidate thresholds with their source (grid and/or PR curve)."""
    grid = threshold_grid(config.grid_start, config.grid_stop, config.grid_step)
    sources: dict[float, set[str]] = {float(t): {SOURCE_GRID} for t in grid}
    if config.include_pr_curve_points:
        _, _, pr_thresholds = precision_recall_curve(np.asarray(y_true), np.asarray(y_proba, dtype=float))
        inside = pr_thresholds[(pr_thresholds > 0.0) & (pr_thresholds < 1.0)]
        dropped = len(pr_thresholds) - len(inside)
        if dropped:
            logger.debug("Dropped %d PR-curve thresholds outside (0, 1)", dropped)
        for t in np.unique(inside):
            sources.setdefault(float(t), set()).add(SOURCE_PR_CURVE)
    thresholds = sorted(sources)
    labels = [SOURCE_BOTH if len(sources[t]) == 2 else next(iter(sources[t])) for t in thresholds]
    return pd.DataFrame({"threshold": thresholds, "source": labels})


def _f_beta(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, beta: float) -> np.ndarray:
    b2 = beta**2
    denom = (1 + b2) * tp + b2 * fn + fp
    return np.divide((1 + b2) * tp, denom, out=np.zeros(len(tp)), where=denom > 0)


def threshold_table(y_true: Any, y_proba: Any, candidates: pd.DataFrame) -> pd.DataFrame:
    """Confusion counts and P/R/F1/F2 at every candidate threshold (vectorised, exact counts)."""
    y = np.asarray(y_true)
    p = np.asarray(y_proba, dtype=float)
    thresholds = candidates["threshold"].to_numpy(dtype=float)
    n, n_pos = len(y), int((y == 1).sum())
    n_neg = n - n_pos

    all_sorted = np.sort(p)
    pos_sorted = np.sort(p[y == 1])
    predicted_positive = n - np.searchsorted(all_sorted, thresholds, side="left")  # p >= t
    tp = n_pos - np.searchsorted(pos_sorted, thresholds, side="left")
    fp = predicted_positive - tp
    fn = n_pos - tp
    tn = n_neg - fp

    precision = np.divide(tp, predicted_positive, out=np.zeros(len(tp)), where=predicted_positive > 0)
    recall = tp / n_pos if n_pos else np.zeros(len(tp))
    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "source": candidates["source"].to_numpy(),
            "predicted_positive": predicted_positive.astype(int),
            "tp": tp.astype(int),
            "fp": fp.astype(int),
            "tn": tn.astype(int),
            "fn": fn.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": _f_beta(tp, fp, fn, 1.0),
            "f2": _f_beta(tp, fp, fn, 2.0),
        }
    )
    return table.loc[:, list(TABLE_COLUMNS)]


def select_threshold(table: pd.DataFrame, r_min: float, grid_start: float, grid_stop: float) -> ThresholdChoice:
    """Apply the documented objective (primary with F2 fallback); ties go to the higher threshold."""
    eligible = table[table["recall"] >= r_min]
    if not eligible.empty:
        objective, pool, key = OBJECTIVE_PRIMARY, eligible, "precision"
    else:
        objective, pool, key = OBJECTIVE_FALLBACK, table, "f2"
    best_value = pool[key].max()
    best = pool[pool[key] == best_value].sort_values("threshold", kind="mergesort").iloc[-1]
    threshold = float(best["threshold"])
    return ThresholdChoice(
        threshold=threshold,
        objective=objective,
        r_min=r_min,
        reached_r_min=not eligible.empty,
        at_grid_edge=threshold <= grid_start or threshold >= grid_stop,
        metrics={k: (int(best[k]) if k in ("predicted_positive", "tp", "fp", "tn", "fn") else best[k])
                 for k in TABLE_COLUMNS if k not in ("threshold",)}
        | {"threshold": threshold},
    )


def analyse_thresholds(
    y_true: Any, y_proba: Any, config: ThresholdConfig
) -> tuple[pd.DataFrame, ThresholdChoice]:
    """Build the threshold table for one model and choose its operating point."""
    table = threshold_table(y_true, y_proba, candidate_thresholds(y_true, y_proba, config))
    choice = select_threshold(table, config.r_min, config.grid_start, config.grid_stop)
    if choice.objective == OBJECTIVE_FALLBACK:
        logger.warning("No threshold reaches recall >= %.2f; F2 fallback applied", config.r_min)
    if choice.at_grid_edge:
        logger.warning("Chosen threshold %.6f is at the grid edge; reported as-is", choice.threshold)
    return table, choice
