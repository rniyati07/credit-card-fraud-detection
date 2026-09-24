"""Leakage guards for preprocessing (DOC-05 M3, §10 L-03 / L-12; DOC-03 §9).

The preprocessor must learn its statistics from the data it is fitted on (train)
and nothing else, and ``Time`` / the target / the row id must never reach it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.split import split_dataset
from fraud_detection.features.preprocess import (
    IMPUTE_STEP,
    ROBUST_GROUP,
    SCALE_STEP,
    STANDARD_GROUP,
    build_preprocessor,
    feature_groups,
    split_features_target,
)


@pytest.fixture
def partitions(make_transactions, schema_config, split_config, temporal_config):
    frame = make_transactions(n_rows=2000, n_fraud=100, seed=3)
    return split_dataset(frame, schema_config, split_config, temporal_config, seed=42).partitions


@pytest.fixture
def fitted(partitions, schema_config, preprocessing_config):
    x_train, _ = split_features_target(partitions["train"], schema_config)
    return build_preprocessor(schema_config, preprocessing_config).fit(x_train), x_train


def _step(preprocessor, group: str, step: str):
    return preprocessor.named_transformers_[group].named_steps[step]


def test_transformer_columns_are_exactly_the_features(schema_config, preprocessing_config) -> None:
    pre = build_preprocessor(schema_config, preprocessing_config)
    columns = [c for _, _, cols in pre.transformers for c in cols]

    assert sorted(columns) == sorted(schema_config.features)
    for forbidden in ("Time", "Class", "row_id"):
        assert forbidden not in columns
    assert pre.remainder == "drop"


def test_amount_is_robust_scaled_others_standard_scaled(schema_config, preprocessing_config) -> None:
    standard, robust = feature_groups(schema_config, preprocessing_config)
    assert robust == ["Amount"]
    assert standard == [f"V{i}" for i in range(1, 29)]


def test_time_as_feature_is_rejected(schema_config, preprocessing_config) -> None:
    bad = dataclasses.replace(schema_config, features=("Time", *schema_config.features[1:]))
    with pytest.raises(ValueError, match="never be a model feature"):
        build_preprocessor(bad, preprocessing_config)


def test_split_features_target_drops_non_features(partitions, schema_config) -> None:
    x, y = split_features_target(partitions["train"], schema_config)

    assert list(x.columns) == list(schema_config.features)
    assert y.name == "Class" and len(y) == len(x)
    with pytest.raises(ValueError, match="missing columns"):
        split_features_target(partitions["train"].drop(columns="V9"), schema_config)


def test_fitted_statistics_equal_train_statistics(fitted, schema_config, preprocessing_config) -> None:
    pre, x_train = fitted
    standard, robust = feature_groups(schema_config, preprocessing_config)

    np.testing.assert_allclose(_step(pre, STANDARD_GROUP, SCALE_STEP).mean_, x_train[standard].mean())
    np.testing.assert_allclose(_step(pre, STANDARD_GROUP, SCALE_STEP).scale_, x_train[standard].std(ddof=0))
    np.testing.assert_allclose(_step(pre, STANDARD_GROUP, IMPUTE_STEP).statistics_, x_train[standard].median())
    np.testing.assert_allclose(_step(pre, ROBUST_GROUP, SCALE_STEP).center_, x_train[robust].median())
    assert list(pre.feature_names_in_) == list(schema_config.features)


def test_transforming_other_splits_does_not_refit(fitted, partitions, schema_config) -> None:
    pre, x_train = fitted
    before = _step(pre, STANDARD_GROUP, SCALE_STEP).mean_.copy()

    for name in ("val", "test", "temporal_holdout"):
        x, _ = split_features_target(partitions[name], schema_config)
        pre.transform(x)

    np.testing.assert_array_equal(_step(pre, STANDARD_GROUP, SCALE_STEP).mean_, before)


def test_train_only_fit_differs_from_fit_on_all_data(fitted, partitions, schema_config, preprocessing_config) -> None:
    # Guard that the statistics comparison above is meaningful: including other
    # partitions (leakage) would produce different scaler parameters.
    pre, _ = fitted
    everything = pd.concat(partitions.values())
    x_all, _ = split_features_target(everything, schema_config)
    leaky = build_preprocessor(schema_config, preprocessing_config).fit(x_all)

    assert not np.allclose(
        _step(pre, STANDARD_GROUP, SCALE_STEP).mean_, _step(leaky, STANDARD_GROUP, SCALE_STEP).mean_
    )


def test_output_shape_and_feature_order(fitted, partitions, schema_config) -> None:
    pre, _ = fitted
    x_val, _ = split_features_target(partitions["val"], schema_config)

    out = pre.transform(x_val)

    assert out.shape == (len(x_val), 29)
    assert list(pre.get_feature_names_out()) == list(schema_config.features)
    np.testing.assert_allclose(out.mean(axis=0)[:28], 0.0, atol=0.5)  # scaled by train stats


def test_missing_value_imputed_with_train_median(fitted, partitions, schema_config) -> None:
    pre, x_train = fitted
    x_val, _ = split_features_target(partitions["val"], schema_config)
    x_val = x_val.head(1).copy()
    x_val.loc[:, "V1"] = np.nan

    out = pre.transform(x_val)

    scaler = _step(pre, STANDARD_GROUP, SCALE_STEP)
    expected = (x_train["V1"].median() - scaler.mean_[0]) / scaler.scale_[0]
    assert out[0, 0] == pytest.approx(expected)


def test_extra_columns_are_ignored(fitted, partitions, schema_config) -> None:
    pre, _ = fitted
    x_val, _ = split_features_target(partitions["val"], schema_config)
    with_extras = partitions["val"]  # includes row_id, Time, Class
    np.testing.assert_array_equal(pre.transform(with_extras), pre.transform(x_val))
