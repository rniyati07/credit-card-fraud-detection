"""Leakage guards for preprocessing and training (DOC-05 M3/M4, §10 L-03 / L-04 / L-06 / L-12; DOC-03 §9).

The preprocessor must learn its statistics from the data it is fitted on (train)
and nothing else, and ``Time`` / the target / the row id must never reach it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_detection.config import load_config
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
from fraud_detection.models.train import (
    MODEL_STEP,
    PREPROCESS_STEP,
    TrainingError,
    build_model_pipeline,
    fit_model_pipeline,
)
from fraud_detection.pipeline import split as split_stage
from fraud_detection.pipeline import train as train_stage
from fraud_detection.pipeline.train import partition_path


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


# ------------------------------------------------------------------ train stage (DOC-05 M4, L-04/L-06)


@pytest.fixture
def split_env(tmp_path, config_dict_factory, transactions_factory):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict_factory(tmp_path)), encoding="utf-8")
    config = load_config(config_path)
    config.paths.interim.parent.mkdir(parents=True)
    transactions_factory(n_rows=2000, n_fraud=100, seed=9, signal=3.0).to_csv(config.paths.interim, index=False)
    assert split_stage.main(["--config", str(config_path), "--log-level", "WARNING"]) == 0
    return config, config_path


def test_train_stage_never_reads_test_or_holdout(split_env, monkeypatch) -> None:
    config, config_path = split_env
    # Structural guard: remove the forbidden files entirely, then also spy on every CSV read.
    for name in ("test.csv", "temporal_holdout.csv"):
        (config.paths.processed_dir / name).unlink()
    read_paths: list[str] = []
    real_read_csv = pd.read_csv

    def spy(path, *args, **kwargs):
        read_paths.append(str(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(train_stage.pd, "read_csv", spy)

    assert train_stage.main(["--config", str(config_path), "--log-level", "WARNING"]) == 0
    assert sorted(Path(p).name for p in read_paths) == ["train.csv", "val.csv"]


def test_partition_path_refuses_test_and_holdout(split_env) -> None:
    config, _ = split_env
    for name in ("test", "temporal_holdout"):
        with pytest.raises(TrainingError, match="must not read"):
            partition_path(config, name)


def test_scale_pos_weight_uses_train_labels_only(partitions, schema_config, preprocessing_config) -> None:
    _, y_train = split_features_target(partitions["train"], schema_config)
    everything = pd.concat([partitions[n] for n in ("train", "val", "test")])
    _, y_all = split_features_target(everything, schema_config)

    pipe = build_model_pipeline("xgboost", {}, schema_config, preprocessing_config, 42, y_train)
    weight = pipe.named_steps[MODEL_STEP].get_params()["scale_pos_weight"]

    assert weight == pytest.approx((y_train == 0).sum() / (y_train == 1).sum())
    assert weight != pytest.approx((y_all == 0).sum() / (y_all == 1).sum())


def test_model_pipeline_fits_preprocessing_on_train_only(partitions, schema_config, preprocessing_config) -> None:
    x_train, y_train = split_features_target(partitions["train"], schema_config)
    pipe = build_model_pipeline(
        "logistic_regression", {"class_weight": "balanced", "max_iter": 1000},
        schema_config, preprocessing_config, 42, y_train,
    )
    fit_model_pipeline(pipe, x_train, y_train)
    x_val, _ = split_features_target(partitions["val"], schema_config)
    pipe.predict_proba(x_val)

    pre = pipe.named_steps[PREPROCESS_STEP]
    standard, _ = feature_groups(schema_config, preprocessing_config)
    np.testing.assert_allclose(_step(pre, STANDARD_GROUP, SCALE_STEP).mean_, x_train[standard].mean())
