"""Inference tests (DOC-04 §4, §6; DOC-05 §15 test_inference.py / test_artifacts.py startup part)."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import fraud_detection.evaluation.evaluate as evaluate_module
import fraud_detection.evaluation.metrics as metrics_module
from conftest import FEATURES, write_serving_artifacts
from fraud_detection.evaluation.threshold import candidate_thresholds, threshold_table
from fraud_detection.inference.predictor import (
    REQUIRED_CONFIG_KEYS,
    FraudPredictor,
    ModelLoadError,
    PredictionError,
    apply_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_THRESHOLD = 0.9111537337303162


class OutOfRangeModel:
    """Picklable stand-in whose probabilities are not probabilities (self-check must fail)."""

    feature_names_in_ = np.array(FEATURES, dtype=object)
    n_features_in_ = len(FEATURES)

    def predict_proba(self, x):
        return np.tile([-1.0, 2.0], (len(x), 1))


def _rewrite_config(config_path: Path, **changes) -> None:
    config = json.loads(config_path.read_text())
    for key, value in changes.items():
        if value is ...:
            config.pop(key)
        else:
            config[key] = value
    config_path.write_text(json.dumps(config))


# ------------------------------------------------------------------ the single decision rule


def test_apply_threshold_uses_greater_or_equal() -> None:
    assert apply_threshold(0.5, 0.5) == 1  # boundary counts as fraud
    assert apply_threshold(0.4999999, 0.5) == 0
    assert apply_threshold(FROZEN_THRESHOLD, FROZEN_THRESHOLD) == 1
    assert apply_threshold(np.nextafter(FROZEN_THRESHOLD, 0), FROZEN_THRESHOLD) == 0
    assert isinstance(apply_threshold(0.9, 0.5), int)


def test_apply_threshold_vectorised() -> None:
    out = apply_threshold([0.1, 0.5, 0.9], 0.5)
    assert out.tolist() == [0, 1, 1] and out.dtype.kind == "i"


def test_evaluation_and_serving_share_one_rule() -> None:
    # One implementation (DOC-04 §6): evaluation modules hold the inference function itself.
    assert metrics_module.apply_threshold is apply_threshold
    assert evaluate_module.apply_threshold is apply_threshold
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "apply_threshold"]
        if path.name != "predictor.py":
            assert not defined, f"apply_threshold redefined in {path}"


def test_m6_vectorised_threshold_counts_match_the_shared_rule() -> None:
    # evaluation/threshold.py counts decisions for thousands of thresholds at once; it must agree
    # with apply_threshold at every candidate threshold.
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(40, int), np.zeros(400, int)]
    p = np.clip(np.r_[rng.normal(0.7, 0.2, 40), rng.normal(0.2, 0.15, 400)], 0, 1)
    from fraud_detection.config import ThresholdConfig

    cfg = ThresholdConfig(r_min=0.8, r_min_confirmed=True, grid_start=0.01, grid_stop=0.99, grid_step=0.01,
                          include_pr_curve_points=True, fallback="f2", reference_threshold=0.5)
    table = threshold_table(y, p, candidate_thresholds(y, p, cfg))
    for row in table.itertuples():
        predicted = apply_threshold(p, row.threshold)
        assert row.tp == int((predicted & y).sum()) and row.fp == int((predicted & (1 - y)).sum())


# ----------------------------------------------------------------------------- loading


def test_load_valid_artifacts(serving_artifacts) -> None:
    model_path, config_path, config = serving_artifacts
    predictor = FraudPredictor.load(model_path, config_path)
    assert predictor.threshold == config["threshold"]
    assert predictor.feature_order == tuple(FEATURES) and "Time" not in predictor.feature_order
    assert predictor.model_version == config["model_version"]
    assert predictor.model_name == config["model_name"]


@pytest.mark.parametrize("which", ["model", "config"])
def test_missing_artifact(serving_artifacts, which) -> None:
    model_path, config_path, _ = serving_artifacts
    (model_path if which == "model" else config_path).unlink()
    with pytest.raises(ModelLoadError, match="Model artifact not found"):
        FraudPredictor.load(model_path, config_path)


def test_config_not_json(serving_artifacts) -> None:
    model_path, config_path, _ = serving_artifacts
    config_path.write_text("{not json")
    with pytest.raises(ModelLoadError, match="Model config invalid"):
        FraudPredictor.load(model_path, config_path)


@pytest.mark.parametrize("key", REQUIRED_CONFIG_KEYS)
def test_missing_required_key(serving_artifacts, key) -> None:
    model_path, config_path, _ = serving_artifacts
    _rewrite_config(config_path, **{key: ...})
    with pytest.raises(ModelLoadError, match=f"Model config invalid: missing {key}"):
        FraudPredictor.load(model_path, config_path)


def test_empty_model_version_rejected(serving_artifacts) -> None:
    model_path, config_path, _ = serving_artifacts
    _rewrite_config(config_path, model_version="")
    with pytest.raises(ModelLoadError, match="model_version"):
        FraudPredictor.load(model_path, config_path)


@pytest.mark.parametrize("threshold", [0, 1, 1.5, -0.1, "0.5", True, None, math.nan])
def test_invalid_threshold(serving_artifacts, threshold) -> None:
    model_path, config_path, _ = serving_artifacts
    config_path.write_text(json.dumps(json.loads(config_path.read_text()) | {"threshold": threshold}))
    with pytest.raises(ModelLoadError, match="Invalid threshold in config"):
        FraudPredictor.load(model_path, config_path)


@pytest.mark.parametrize(
    "feature_order",
    [
        ["Time", *FEATURES[1:]],          # Time is never a feature
        [*FEATURES[:-1], "Class"],        # nor the target
        FEATURES[:-1],                    # 28 features
        [*FEATURES[:-1], FEATURES[0]],    # duplicate
        "V1,V2",                          # not a list
    ],
)
def test_bad_feature_order(serving_artifacts, feature_order) -> None:
    model_path, config_path, _ = serving_artifacts
    _rewrite_config(config_path, feature_order=feature_order)
    with pytest.raises(ModelLoadError, match="Feature schema incompatible"):
        FraudPredictor.load(model_path, config_path)


def test_feature_order_must_match_model_inputs(serving_artifacts) -> None:
    model_path, config_path, _ = serving_artifacts
    _rewrite_config(config_path, feature_order=list(reversed(FEATURES)))  # 29 valid names, wrong order
    with pytest.raises(ModelLoadError, match="feature_names_in_"):
        FraudPredictor.load(model_path, config_path)


def test_library_version_mismatch(serving_artifacts) -> None:
    model_path, config_path, config = serving_artifacts
    _rewrite_config(config_path, library_versions=config["library_versions"] | {"xgboost": "0.0.1"})
    with pytest.raises(ModelLoadError, match="Library version mismatch: xgboost"):
        FraudPredictor.load(model_path, config_path)


def test_corrupt_model(serving_artifacts) -> None:
    model_path, config_path, _ = serving_artifacts
    model_path.write_bytes(b"not a pickle")
    with pytest.raises(ModelLoadError, match="Model artifact corrupt or incompatible"):
        FraudPredictor.load(model_path, config_path)


def test_object_without_predict_proba(serving_artifacts) -> None:
    model_path, config_path, _ = serving_artifacts
    joblib.dump({"not": "a model"}, model_path)
    with pytest.raises(ModelLoadError, match="no predict_proba"):
        FraudPredictor.load(model_path, config_path)


def test_self_check_rejects_non_probabilities(serving_artifacts) -> None:
    model_path, config_path, _ = serving_artifacts
    joblib.dump(OutOfRangeModel(), model_path)
    with pytest.raises(ModelLoadError, match="Model self-check failed"):
        FraudPredictor.load(model_path, config_path)


# ------------------------------------------------------------------------------ predicting


def test_frame_follows_feature_order(serving_artifacts, transaction) -> None:
    predictor = FraudPredictor.load(*serving_artifacts[:2])
    shuffled = dict(reversed(list(transaction.items())))
    frame = predictor.build_frame(shuffled)
    assert list(frame.columns) == list(predictor.feature_order)
    assert frame.iloc[0].tolist() == [transaction[f] for f in predictor.feature_order]


@pytest.mark.parametrize("change", ["missing", "Time", "extra"])
def test_frame_rejects_schema_mismatch(serving_artifacts, transaction, change) -> None:
    predictor = FraudPredictor.load(*serving_artifacts[:2])
    features = dict(transaction)
    if change == "missing":
        features.pop("V7")
    else:
        features["Time" if change == "Time" else "foo"] = 1.0
    with pytest.raises(ValueError, match="feature_order"):
        predictor.build_frame(features)


def test_predict_applies_frozen_threshold(serving_artifacts, transaction) -> None:
    model_path, config_path, config = serving_artifacts
    predictor = FraudPredictor.load(model_path, config_path)
    result = predictor.predict(transaction)
    pipeline = joblib.load(model_path)
    expected = pipeline.predict_proba(pd.DataFrame([transaction])[FEATURES])[0, 1]

    assert result.fraud_probability == pytest.approx(expected)
    assert 0.0 <= result.fraud_probability <= 1.0
    assert result.threshold == config["threshold"]
    assert result.prediction == int(result.fraud_probability >= result.threshold)
    assert result.label == ("fraud" if result.prediction else "legitimate")
    assert result.model_version == config["model_version"]


def test_decision_flips_exactly_at_threshold(tmp_path, transaction) -> None:
    probe = write_serving_artifacts(tmp_path / "probe")
    p = FraudPredictor.load(*probe[:2]).predict(transaction).fraud_probability
    at = FraudPredictor.load(*write_serving_artifacts(tmp_path / "at", threshold=p)[:2]).predict(transaction)
    above = FraudPredictor.load(
        *write_serving_artifacts(tmp_path / "above", threshold=float(np.nextafter(p, 1)))[:2]
    ).predict(transaction)
    assert at.prediction == 1 and above.prediction == 0


def test_key_order_does_not_change_prediction(serving_artifacts, transaction) -> None:
    predictor = FraudPredictor.load(*serving_artifacts[:2])
    assert predictor.predict(transaction) == predictor.predict(dict(reversed(list(transaction.items()))))


def test_model_output_outside_unit_interval_is_an_error(serving_artifacts, transaction, monkeypatch) -> None:
    predictor = FraudPredictor.load(*serving_artifacts[:2])
    monkeypatch.setattr(predictor, "predict_proba", lambda frame: np.array([1.5]))
    with pytest.raises(PredictionError):
        predictor.predict(transaction)


# --------------------------------------------------------------------------- guarantees


def test_serving_code_imports_no_training_modules() -> None:
    # DOC-03 §11: inference/ and app/ must not import data, models, evaluation, tracking, pipeline
    # (nor config/features, which the serving image does not ship, DOC-04 §8).
    forbidden = ("data", "models", "evaluation", "tracking", "pipeline", "features", "config", "eda")
    files = [*(ROOT / "src" / "fraud_detection" / "inference").glob("*.py"), *(ROOT / "app").glob("*.py")]
    assert files
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for module in modules:
                parts = module.split(".")
                if parts[0] == "fraud_detection" and len(parts) > 1:
                    assert parts[1] == "inference", f"{path.name} imports {module}"
                assert parts[0] not in ("sklearn", "xgboost", "mlflow", "dvc"), f"{path.name} imports {module}"
                assert not (parts[0] == "fraud_detection" and len(parts) > 1 and parts[1] in forbidden)


def test_loading_and_predicting_never_fit_or_write(serving_artifacts, transaction, monkeypatch) -> None:
    model_path, config_path, _ = serving_artifacts
    before = {model_path: model_path.read_bytes(), config_path: config_path.read_bytes()}

    def forbidden(*args, **kwargs):
        raise AssertionError("fit called during serving")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    predictor = FraudPredictor.load(model_path, config_path)
    for _ in range(3):
        predictor.predict(transaction)
    assert {p: p.read_bytes() for p in before} == before


# ------------------------------------------------------------- real artifacts (optional)

REAL_MODEL = ROOT / "models" / "model.joblib"
REAL_CONFIG = ROOT / "models" / "model_config.json"
REAL_TEST = ROOT / "data" / "processed" / "test.csv"


@pytest.mark.artifact
@pytest.mark.skipif(not (REAL_MODEL.exists() and REAL_CONFIG.exists() and REAL_TEST.exists()),
                    reason="real DVC outputs not present (run `dvc checkout` or `dvc repro`)")
def test_real_frozen_model_parity() -> None:
    predictor = FraudPredictor.load(REAL_MODEL, REAL_CONFIG)
    assert predictor.model_name == "xgboost"
    assert predictor.threshold == FROZEN_THRESHOLD
    rows = pd.read_csv(REAL_TEST).head(50)
    offline = joblib.load(REAL_MODEL).predict_proba(rows[list(predictor.feature_order)])[:, 1]
    for i, (_, row) in enumerate(rows.iterrows()):
        result = predictor.predict({f: float(row[f]) for f in predictor.feature_order})
        assert result.fraud_probability == pytest.approx(offline[i], abs=1e-12)
        assert result.prediction == apply_threshold(offline[i], FROZEN_THRESHOLD)
