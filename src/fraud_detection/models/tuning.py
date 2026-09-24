"""Train-only cross-validated hyperparameter search (DOC-02 §10, TD-09; DOC-05 M5).

* The search object receives the **train split only**. Validation data never influences
  which hyperparameters are chosen (DOC-05 M5 "Must NOT tune on validation").
* The full model ``Pipeline`` (preprocessing + estimator) is searched, so preprocessing
  is refitted inside every CV fold on that fold's training part only.
* ``StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)``,
  ``scoring="average_precision"`` (PR-AUC).
* Spaces with at most ``n_iter`` combinations (the Logistic Regression grid) are searched
  exhaustively with ``GridSearchCV``; larger spaces use ``RandomizedSearchCV`` with
  ``n_iter`` seeded samples.
* ``refit=True``: the best parameters are refitted on the full train split.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from fraud_detection.config import PreprocessingConfig, SchemaConfig, TuningConfig
from fraud_detection.models.train import MODEL_STEP, build_model_pipeline

logger = logging.getLogger(__name__)

PARAM_PREFIX = f"{MODEL_STEP}__"
GRID = "grid"
RANDOMIZED = "randomized"


class TuningError(Exception):
    """Raised when a search cannot be built or run."""


@dataclass(frozen=True)
class SearchResult:
    """Outcome of one model's search: best params, CV scores, all trials, refitted pipeline."""

    model: str
    strategy: str
    n_candidates: int
    best_params: dict[str, Any]
    cv_best_score: float
    cv_best_std: float
    trials: pd.DataFrame
    best_pipeline: Pipeline


def search_space_size(space: dict[str, tuple[Any, ...]]) -> int:
    """Number of distinct parameter combinations in a list-valued space."""
    return math.prod(len(values) for values in space.values())


def make_cv(cv_folds: int, seed: int) -> StratifiedKFold:
    """The documented CV splitter: stratified, shuffled, seeded."""
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)


def build_search(
    pipeline: Pipeline, space: dict[str, tuple[Any, ...]], tuning: TuningConfig, seed: int
) -> GridSearchCV | RandomizedSearchCV:
    """Grid search when the space fits within ``n_iter``, otherwise seeded randomized search.

    Raises:
        TuningError: If a searched parameter does not exist on the estimator.
    """
    grid = {f"{PARAM_PREFIX}{name}": list(values) for name, values in space.items()}
    unknown = sorted(set(grid) - set(pipeline.get_params()))
    if unknown:
        raise TuningError(f"Unknown hyperparameters for this estimator: {[u.removeprefix(PARAM_PREFIX) for u in unknown]}")

    common: dict[str, Any] = {
        "scoring": tuning.scoring,
        "cv": make_cv(tuning.cv_folds, seed),
        "refit": True,
        "n_jobs": 1,  # estimators parallelise internally; avoids nested oversubscription
        "error_score": "raise",
        "return_train_score": False,
    }
    if search_space_size(space) <= tuning.n_iter:
        return GridSearchCV(pipeline, grid, **common)
    return RandomizedSearchCV(pipeline, grid, n_iter=tuning.n_iter, random_state=seed, **common)


def cv_results_frame(
    model: str, cv_results: dict[str, Any], param_names: list[str], cv_folds: int
) -> pd.DataFrame:
    """Deterministic per-trial table (no fit/score timings), ordered by rank then trial."""
    rows = []
    for i, params in enumerate(cv_results["params"]):
        plain = {k.removeprefix(PARAM_PREFIX): v for k, v in params.items()}
        row: dict[str, Any] = {
            "model": model,
            "trial": i,
            "rank": int(cv_results["rank_test_score"][i]),
            "params": json.dumps(plain, sort_keys=True),
        }
        row |= {f"param_{p}": plain.get(p) for p in param_names}
        row["mean_cv_average_precision"] = float(cv_results["mean_test_score"][i])
        row["std_cv_average_precision"] = float(cv_results["std_test_score"][i])
        row |= {
            f"split{k}_cv_average_precision": float(cv_results[f"split{k}_test_score"][i])
            for k in range(cv_folds)
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rank", "trial"], kind="mergesort").reset_index(drop=True)


def run_search(
    name: str,
    fixed_params: dict[str, Any],
    space: dict[str, tuple[Any, ...]],
    tuning: TuningConfig,
    schema: SchemaConfig,
    preprocessing: PreprocessingConfig,
    seed: int,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> SearchResult:
    """Search one candidate on the train split and refit the best parameters on all of it.

    Raises:
        TuningError: If the search cannot be built or fitting fails.
    """
    pipeline = build_model_pipeline(name, fixed_params, schema, preprocessing, seed, y_train)
    search = build_search(pipeline, space, tuning, seed)
    strategy = GRID if isinstance(search, GridSearchCV) else RANDOMIZED
    n_candidates = search_space_size(space) if strategy == GRID else tuning.n_iter
    logger.info(
        "Tuning %s: %s search, %d candidates x %d folds on %d train rows",
        name, strategy, n_candidates, tuning.cv_folds, len(x_train),
    )
    try:
        search.fit(x_train, y_train)
    except ValueError as exc:
        raise TuningError(f"Search for '{name}' failed: {exc}") from exc

    best = search.best_index_
    best_params = {k.removeprefix(PARAM_PREFIX): v for k, v in search.best_params_.items()}
    result = SearchResult(
        model=name,
        strategy=strategy,
        n_candidates=n_candidates,
        best_params=dict(sorted(best_params.items())),
        cv_best_score=float(search.best_score_),
        cv_best_std=float(search.cv_results_["std_test_score"][best]),
        trials=cv_results_frame(name, search.cv_results_, list(space), tuning.cv_folds),
        best_pipeline=search.best_estimator_,
    )
    logger.info(
        "%s best CV average precision %.4f (+/- %.4f) with %s",
        name, result.cv_best_score, result.cv_best_std, result.best_params,
    )
    return result
