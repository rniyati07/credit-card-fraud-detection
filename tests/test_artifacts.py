"""Artifact tests (DOC-05 §15 test_artifacts.py, M7 part; DOC-01 SC-08).

The frozen pair ``model.joblib`` + ``model_config.json`` must load and reproduce the test
predictions and metrics, and the threshold must be identical in ``train_metadata.json`` and
``model_config.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_detection.config import load_config
from fraud_detection.evaluation.metrics import threshold_metrics
from fraud_detection.features.preprocess import split_features_target
from fraud_detection.pipeline import evaluate, evaluate_temporal, select, split, tune


@pytest.fixture(scope="module")
def frozen(tmp_path_factory, config_dict_factory, transactions_factory):
    tmp_path = tmp_path_factory.mktemp("artifacts")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict_factory(tmp_path)), encoding="utf-8")
    config = load_config(config_path)
    config.paths.interim.parent.mkdir(parents=True)
    transactions_factory(n_rows=3000, n_fraud=150, seed=11, signal=3.0).to_csv(config.paths.interim, index=False)
    for stage in (split, tune, select, evaluate, evaluate_temporal):
        assert stage.main(["--config", str(config_path), "--log-level", "ERROR"]) == 0
    return config


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_threshold_identical_across_metadata_and_model_config(frozen) -> None:
    metadata = _json(frozen.paths.models_dir / "train_metadata.json")
    model_config = _json(frozen.paths.models_dir / "model_config.json")
    assert model_config["threshold"] == metadata["threshold"]  # exact, copied verbatim
    assert model_config["feature_order"] == metadata["feature_order"]
    assert model_config["model_version"] == metadata["model_version"]
    assert model_config["model_sha256"] == metadata["model_sha256"]


def test_save_load_round_trip_gives_identical_probabilities(frozen, tmp_path) -> None:
    x, _ = split_features_target(pd.read_csv(frozen.paths.processed_dir / "test.csv"), frozen.schema)
    original = joblib.load(frozen.paths.models_dir / "model.joblib")
    copy_path = tmp_path / "model.joblib"
    joblib.dump(original, copy_path)
    np.testing.assert_array_equal(original.predict_proba(x), joblib.load(copy_path).predict_proba(x))


def test_frozen_pair_reproduces_test_metrics(frozen) -> None:
    model_config = _json(frozen.paths.models_dir / "model_config.json")
    test_metrics = _json(frozen.paths.test_metrics)
    x, y = split_features_target(pd.read_csv(frozen.paths.processed_dir / "test.csv"), frozen.schema)
    proba = joblib.load(frozen.paths.models_dir / "model.joblib").predict_proba(x[model_config["feature_order"]])[:, 1]

    reproduced = threshold_metrics(y, proba, model_config["threshold"])
    for key in ("tp", "fp", "tn", "fn"):
        assert reproduced[key] == test_metrics["at_threshold"][key] == model_config["test_metrics"]["at_threshold"][key]
