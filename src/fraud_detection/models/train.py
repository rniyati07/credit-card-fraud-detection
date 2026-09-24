"""Estimator and model-pipeline factories for the three candidates (DOC-02 §8-§9, DOC-05 M4).

Each model is ``Pipeline([("preprocess", ColumnTransformer), ("model", estimator)])``
so preprocessing is fitted together with the model, on the train split only.

Imbalance handling is cost-sensitive (TD-06):

* Logistic Regression: ``class_weight="balanced"`` (from config)
* Random Forest: ``class_weight="balanced_subsample"`` (from config)
* XGBoost: ``scale_pos_weight = n_negative_train / n_positive_train``, computed here
  from the **train labels only** (L-06)

``random_state`` is always the global project seed (NFR-002).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_detection.config import CANDIDATE_MODELS, PreprocessingConfig, SchemaConfig
from fraud_detection.features.preprocess import build_preprocessor

logger = logging.getLogger(__name__)

PREPROCESS_STEP = "preprocess"
MODEL_STEP = "model"


class TrainingError(Exception):
    """Raised when a model cannot be built or trained on the given data."""


def compute_scale_pos_weight(y_train: Any, positive_label: int = 1) -> float:
    """``n_negative / n_positive`` of the **train** labels (XGBoost imbalance weight).

    Raises:
        TrainingError: If the labels contain no positive or no negative example.
    """
    y = np.asarray(y_train)
    n_pos = int(np.sum(y == positive_label))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise TrainingError(f"Train labels need both classes (positives={n_pos}, negatives={n_neg})")
    return n_neg / n_pos


def build_estimator(
    name: str, params: dict[str, Any], seed: int, y_train: Any, positive_label: int = 1
) -> BaseEstimator:
    """Create an unfitted estimator from configured fixed params plus code-injected values.

    Raises:
        TrainingError: For an unknown model name.
    """
    params = dict(params)
    if name == "logistic_regression":
        return LogisticRegression(**params, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(**params, random_state=seed)
    if name == "xgboost":
        weight = compute_scale_pos_weight(y_train, positive_label)
        return XGBClassifier(**params, scale_pos_weight=weight, random_state=seed)
    raise TrainingError(f"Unknown model '{name}'; allowed: {list(CANDIDATE_MODELS)}")


def build_model_pipeline(
    name: str,
    params: dict[str, Any],
    schema: SchemaConfig,
    preprocessing: PreprocessingConfig,
    seed: int,
    y_train: Any,
) -> Pipeline:
    """Unfitted ``Pipeline([("preprocess", ...), ("model", ...)])`` for one candidate."""
    return Pipeline(
        [
            (PREPROCESS_STEP, build_preprocessor(schema, preprocessing)),
            (MODEL_STEP, build_estimator(name, params, seed, y_train, schema.positive_label)),
        ]
    )


def fit_model_pipeline(pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit preprocessing and model on the train split only."""
    name = type(pipeline.named_steps[MODEL_STEP]).__name__
    logger.info("Fitting %s on %d train rows (%d frauds)", name, len(x_train), int(y_train.sum()))
    return pipeline.fit(x_train, y_train)


def predict_proba(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    """Fraud probability ``P(Class = 1 | x)`` for every row, bit-for-bit reproducible.

    A multi-threaded ``RandomForestClassifier`` sums per-tree probabilities in thread
    completion order, which changes the last bit of a few probabilities between calls.
    Forests are therefore scored single-threaded here; the fitted object is not modified.
    """
    model = pipeline.named_steps[MODEL_STEP]
    if isinstance(model, RandomForestClassifier) and model.n_jobs not in (None, 1):
        original = model.n_jobs
        model.n_jobs = 1
        try:
            return pipeline.predict_proba(x)[:, 1]
        finally:
            model.n_jobs = original
    return pipeline.predict_proba(x)[:, 1]


def model_params_for_logging(pipeline: Pipeline) -> dict[str, Any]:
    """Resolved estimator hyperparameters as plain scalars (for MLflow and reports)."""
    params = pipeline.named_steps[MODEL_STEP].get_params()
    out: dict[str, Any] = {}
    for key, value in sorted(params.items()):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = str(value)  # e.g. XGBoost missing=nan; keeps JSON strict
        if value is None or isinstance(value, (bool, int, float, str)):
            out[key] = value
    return out
