"""Frozen-model inference and the project's single decision rule (FR-016; DOC-04 §4, §6).

* :func:`apply_threshold` is the **only** implementation of the decision rule
  ``fraud iff probability >= threshold``. Evaluation (M7) imports it from here, so evaluation and
  serving cannot diverge (DOC-04 §6).
* :meth:`FraudPredictor.load` loads ``model.joblib`` + ``model_config.json`` once and runs the
  DOC-04 §4 fail-fast startup checks. Nothing is refitted, retuned or written back.
* Inputs are ordered strictly by ``model_config["feature_order"]``; the service performs no
  transformation of its own (scaling/imputation live inside the pipeline).

This module imports only the standard library, numpy, pandas and joblib: no training,
evaluation, tracking, pipeline or config code (DOC-03 §11 import rule).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_N_FEATURES = 29
FORBIDDEN_FEATURES = ("Time", "Class")  # ordering key and target are never model inputs (NFR-010)
REQUIRED_CONFIG_KEYS = ("model_name", "model_version", "threshold", "feature_order", "library_versions")
STRICT_LIBRARIES = ("scikit-learn", "xgboost")  # exact version match required (DOC-04 §4 check 5, DD-07)
LABELS = {0: "legitimate", 1: "fraud"}


class ModelLoadError(RuntimeError):
    """A frozen artifact is missing, corrupt or incompatible; the service must not start."""


class PredictionError(RuntimeError):
    """The model produced an unusable output for a request."""


def apply_threshold(probability: Any, threshold: float) -> Any:
    """The project decision rule: ``1`` (fraud) iff ``probability >= threshold``, else ``0``.

    Accepts a scalar (returns ``int``) or an array (returns an ``int`` numpy array).
    """
    decision = np.asarray(probability, dtype=float) >= threshold
    if decision.ndim == 0:
        return int(decision)
    return decision.astype(int)


@dataclass(frozen=True)
class Prediction:
    """One scored transaction (DOC-04 §5.2 response fields)."""

    fraud_probability: float
    prediction: int
    label: str
    threshold: float
    model_name: str
    model_version: str


# ---------------------------------------------------------------------------- startup checks


def _read_config(config_path: Path) -> dict[str, Any]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"Model config invalid: cannot parse {config_path.name} ({type(exc).__name__})") from exc
    if not isinstance(config, dict):
        raise ModelLoadError("Model config invalid: top level must be a JSON object")
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise ModelLoadError(f"Model config invalid: missing {key}")
    for key in ("model_name", "model_version"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ModelLoadError(f"Model config invalid: {key} must be a non-empty string")
    if not isinstance(config["library_versions"], dict):
        raise ModelLoadError("Model config invalid: library_versions must be an object")
    return config


def _check_feature_order(feature_order: Any) -> tuple[str, ...]:
    if not isinstance(feature_order, list) or not all(isinstance(f, str) for f in feature_order):
        raise ModelLoadError("Feature schema incompatible: feature_order must be a list of names")
    if len(feature_order) != EXPECTED_N_FEATURES or len(set(feature_order)) != EXPECTED_N_FEATURES:
        raise ModelLoadError(
            f"Feature schema incompatible: expected {EXPECTED_N_FEATURES} unique features, got {len(feature_order)}"
        )
    forbidden = [f for f in FORBIDDEN_FEATURES if f in feature_order]
    if forbidden:
        raise ModelLoadError(f"Feature schema incompatible: {forbidden} must never be a model feature")
    return tuple(feature_order)


def _check_threshold(threshold: Any) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ModelLoadError("Invalid threshold in config: must be a number")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ModelLoadError("Invalid threshold in config: must satisfy 0 < threshold < 1")
    return float(threshold)


def _check_library_versions(library_versions: Mapping[str, Any]) -> None:
    for library in STRICT_LIBRARIES:
        try:
            runtime = metadata.version(library)
        except metadata.PackageNotFoundError as exc:
            raise ModelLoadError(f"Library version mismatch: {library} (not installed)") from exc
        frozen = library_versions.get(library)
        if frozen != runtime:
            raise ModelLoadError(f"Library version mismatch: {library} (model {frozen}, runtime {runtime})")


def _load_pipeline(model_path: Path) -> Any:
    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:  # corrupt pickles raise many exception types
        raise ModelLoadError(f"Model artifact corrupt or incompatible ({type(exc).__name__})") from exc
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ModelLoadError("Model artifact corrupt or incompatible: no predict_proba")
    return pipeline


def _check_model_inputs(pipeline: Any, feature_order: tuple[str, ...]) -> None:
    names = getattr(pipeline, "feature_names_in_", None)
    if names is not None and list(names) != list(feature_order):
        raise ModelLoadError("Feature schema incompatible: model feature_names_in_ differ from feature_order")
    n_features = getattr(pipeline, "n_features_in_", None)
    if n_features is not None and int(n_features) != len(feature_order):
        raise ModelLoadError(
            f"Feature schema incompatible: model expects {n_features} inputs, feature_order has {len(feature_order)}"
        )


# ------------------------------------------------------------------------------ predictor


class FraudPredictor:
    """Frozen pipeline + frozen threshold. Create with :meth:`load`; read-only afterwards."""

    def __init__(self, pipeline: Any, config: Mapping[str, Any], feature_order: tuple[str, ...], threshold: float):
        self._pipeline = pipeline
        self._config = dict(config)
        self.feature_order = feature_order
        self.threshold = threshold
        self.model_name: str = config["model_name"]
        self.model_version: str = config["model_version"]

    @classmethod
    def load(cls, model_path: str | Path, config_path: str | Path) -> FraudPredictor:
        """Load the artifact pair and run the DOC-04 §4 startup checks in order.

        Raises:
            ModelLoadError: On the first failed check, with the DOC-04 §4 message.
        """
        model_path, config_path = Path(model_path), Path(config_path)
        for path in (model_path, config_path):  # check 1
            if not path.is_file():
                raise ModelLoadError(f"Model artifact not found at {path}")
        config = _read_config(config_path)  # check 2
        feature_order = _check_feature_order(config["feature_order"])  # check 3 (schema field set: app)
        threshold = _check_threshold(config["threshold"])  # check 4
        _check_library_versions(config["library_versions"])  # check 5
        pipeline = _load_pipeline(model_path)  # check 6
        _check_model_inputs(pipeline, feature_order)  # check 7
        predictor = cls(pipeline, config, feature_order, threshold)
        predictor._self_check()  # check 8
        logger.info(
            "Loaded frozen model %s (version %s), threshold %s", predictor.model_name,
            predictor.model_version, predictor.threshold,
        )
        return predictor

    def _self_check(self) -> None:
        """Check 8: a zero vector in feature_order must score a probability in [0, 1]."""
        frame = pd.DataFrame(np.zeros((1, len(self.feature_order))), columns=list(self.feature_order))
        try:
            probability = self.predict_proba(frame)
        except Exception as exc:
            raise ModelLoadError(f"Model self-check failed ({type(exc).__name__})") from exc
        if probability.shape != (1,) or not np.all(np.isfinite(probability)) or not 0.0 <= probability[0] <= 1.0:
            raise ModelLoadError("Model self-check failed: probability outside [0, 1]")

    def build_frame(self, features: Mapping[str, float]) -> pd.DataFrame:
        """One-row DataFrame with columns exactly ``feature_order`` (input key order is irrelevant).

        Raises:
            ValueError: If a feature is missing or an unknown key is present.
        """
        missing = [f for f in self.feature_order if f not in features]
        extra = sorted(set(features) - set(self.feature_order))
        if missing or extra:
            raise ValueError(f"Input does not match feature_order (missing: {missing}, unexpected: {extra})")
        row = [[float(features[f]) for f in self.feature_order]]
        return pd.DataFrame(row, columns=list(self.feature_order), dtype=float)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Fraud probability ``predict_proba(X)[:, 1]`` for each row of ``frame``."""
        return np.asarray(self._pipeline.predict_proba(frame[list(self.feature_order)])[:, 1], dtype=float)

    def predict(self, features: Mapping[str, float]) -> Prediction:
        """Score one transaction with the frozen model and frozen threshold.

        Raises:
            ValueError: If the input does not match ``feature_order``.
            PredictionError: If the model output is not a probability.
        """
        probability = float(self.predict_proba(self.build_frame(features))[0])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise PredictionError("Model returned a value outside [0, 1]")
        decision = apply_threshold(probability, self.threshold)
        return Prediction(
            fraud_probability=probability,
            prediction=decision,
            label=LABELS[decision],
            threshold=self.threshold,
            model_name=self.model_name,
            model_version=self.model_version,
        )
