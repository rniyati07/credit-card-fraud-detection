"""Shared preprocessing definition (DOC-02 §7, TD-04 / TD-04a).

One unfitted ``ColumnTransformer`` serves all candidate models, so training and serving
apply identical transformations. It is fitted later, on the **train split only**, as the
first step of each model ``Pipeline``. Nothing is fitted in this module.

============================  ==============================================
Feature group                 Transform
============================  ==============================================
``V1``-``V28`` (and any       ``SimpleImputer(median)`` -> ``StandardScaler``
other non-robust feature)
``Amount``                    ``SimpleImputer(median)`` -> ``RobustScaler``
``Time``, ``Class``, row id   never passed in (``remainder="drop"``)
============================  ==============================================

The median imputer is defensive: the clean data has no missing values, but it keeps
inference robust.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from fraud_detection.config import EXPECTED_N_FEATURES, PreprocessingConfig, SchemaConfig

logger = logging.getLogger(__name__)

STANDARD_GROUP = "standard"
ROBUST_GROUP = "robust"
IMPUTE_STEP = "impute"
SCALE_STEP = "scale"


def check_feature_invariants(schema: SchemaConfig) -> None:
    """Assert the non-configurable feature invariants (DOC-05 §7).

    Raises:
        ValueError: If the ordering column or target is a feature, or the count is not 29.
    """
    features = set(schema.features)
    if schema.ordering_column in features:
        raise ValueError(f"'{schema.ordering_column}' must never be a model feature")
    if schema.target in features:
        raise ValueError(f"Target '{schema.target}' must never be a model feature")
    if len(schema.features) != EXPECTED_N_FEATURES or len(features) != EXPECTED_N_FEATURES:
        raise ValueError(f"Expected exactly {EXPECTED_N_FEATURES} unique features")


def feature_groups(
    schema: SchemaConfig, preprocessing: PreprocessingConfig
) -> tuple[list[str], list[str]]:
    """Return ``(standard_scaled, robust_scaled)`` feature lists, each in feature order."""
    robust = [f for f in schema.features if f in preprocessing.robust_scaled_features]
    standard = [f for f in schema.features if f not in preprocessing.robust_scaled_features]
    return standard, robust


def _scaling_pipeline(scaler: StandardScaler | RobustScaler) -> Pipeline:
    return Pipeline([(IMPUTE_STEP, SimpleImputer(strategy="median")), (SCALE_STEP, scaler)])


def build_preprocessor(
    schema: SchemaConfig, preprocessing: PreprocessingConfig
) -> ColumnTransformer:
    """Build the unfitted ``ColumnTransformer`` over the 29 configured features.

    Columns not listed in ``schema.features`` (``Time``, ``Class``, the row id) are
    dropped. Output columns keep their original names (standard group first, then robust).
    """
    check_feature_invariants(schema)
    standard, robust = feature_groups(schema, preprocessing)
    transformers = [
        (STANDARD_GROUP, _scaling_pipeline(StandardScaler()), standard),
        (ROBUST_GROUP, _scaling_pipeline(RobustScaler()), robust),
    ]
    logger.debug("Preprocessor: %d standard-scaled, %d robust-scaled features", len(standard), len(robust))
    return ColumnTransformer(
        [t for t in transformers if t[2]],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_features_target(df: pd.DataFrame, schema: SchemaConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model inputs ``X`` (exactly the configured features, in order) and target ``y``.

    Raises:
        ValueError: If a feature or the target column is missing.
    """
    check_feature_invariants(schema)
    missing = [c for c in (*schema.features, schema.target) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df.loc[:, list(schema.features)].copy(), df[schema.target].copy()
