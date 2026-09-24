"""Final evaluation of the frozen model on one dataset (DOC-02 §14, §21.5; DOC-05 M7).

Nothing is fitted, tuned or re-selected here. The frozen pipeline (``models/model.joblib``)
and threshold (``models/train_metadata.json``) are loaded from disk and verified:

* the model file's SHA-256 equals ``train_metadata.model_sha256`` (round-trip integrity);
* its input features equal ``feature_order`` and the configured features;
* ``0 < threshold < 1``;
* the runtime scikit-learn / xgboost versions equal ``library_versions`` (DOC-04 §4 check 5).

Metrics use the project decision rule ``fraud iff p >= threshold`` at the frozen threshold,
with the 0.50 threshold reported for reference only.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score

from fraud_detection.config import Config, EvaluationConfig, SchemaConfig
from fraud_detection.data.split import PARTITION_FILENAMES
from fraud_detection.evaluation.metrics import compute_metrics, threshold_metrics
from fraud_detection.inference.predictor import apply_threshold
from fraud_detection.tracking.mlflow_utils import file_sha256

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "train_metadata.json"
MODEL_CONFIG_FILENAME = "model_config.json"
REQUIRED_METADATA_KEYS = (
    "model_name", "model_version", "threshold", "threshold_objective", "r_min", "feature_order",
    "model_sha256", "library_versions", "mlflow_run_id", "validation_metrics",
)
STRICT_LIBRARIES = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}


class EvaluationError(Exception):
    """Raised when frozen artifacts or evaluation inputs are missing, altered or incompatible."""


@dataclass(frozen=True)
class FrozenModel:
    """The frozen pipeline plus its metadata (threshold, identity, lineage)."""

    pipeline: Any
    metadata: dict[str, Any]
    model_path: Path
    metadata_path: Path

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])


# ------------------------------------------------------------------------------ loading


def load_train_metadata(path: Path) -> dict[str, Any]:
    """Read ``train_metadata.json`` and check the keys evaluation relies on."""
    if not path.is_file():
        raise EvaluationError(
            f"Frozen metadata not found at '{path}'. Run `python -m fraud_detection.pipeline.select` first."
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Frozen metadata at '{path}' is not valid JSON: {exc}") from exc
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in metadata]
    if missing:
        raise EvaluationError(f"Frozen metadata is missing keys: {missing}")
    return metadata


def load_frozen_model(config: Config) -> FrozenModel:
    """Load and verify the frozen model and threshold (never refits anything).

    Raises:
        EvaluationError: If an artifact is missing, altered or incompatible.
    """
    models_dir = config.paths.models_dir
    metadata_path = models_dir / METADATA_FILENAME
    model_path = models_dir / MODEL_FILENAME
    metadata = load_train_metadata(metadata_path)
    if not model_path.is_file():
        raise EvaluationError(f"Frozen model not found at '{model_path}'")
    if file_sha256(model_path) != metadata["model_sha256"]:
        raise EvaluationError(
            f"'{model_path}' does not match train_metadata.json (SHA-256 differs); the frozen pair is inconsistent"
        )

    threshold = metadata["threshold"]
    if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) < 1.0:
        raise EvaluationError(f"Invalid frozen threshold {threshold!r}; expected 0 < threshold < 1")
    if metadata["feature_order"] != list(config.schema.features):
        raise EvaluationError("train_metadata feature_order differs from the configured features")
    for library, runtime in STRICT_LIBRARIES.items():
        frozen = metadata["library_versions"].get(library)
        if frozen != runtime:
            raise EvaluationError(f"Library version mismatch: {library} frozen {frozen}, runtime {runtime}")

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:  # corrupt pickles raise many exception types
        raise EvaluationError(f"Frozen model at '{model_path}' could not be loaded: {exc}") from exc
    if not hasattr(pipeline, "predict_proba"):
        raise EvaluationError("Frozen model does not provide predict_proba")
    if list(getattr(pipeline, "feature_names_in_", [])) != metadata["feature_order"]:
        raise EvaluationError("Frozen model was fitted on a different feature order")
    logger.info(
        "Loaded frozen %s (version %s), threshold %.10f", metadata["model_name"],
        metadata["model_version"], float(threshold),
    )
    return FrozenModel(pipeline, metadata, model_path, metadata_path)


def load_evaluation_partition(config: Config, name: str) -> tuple[pd.DataFrame, Path]:
    """Load one evaluation partition (``test`` or ``temporal_holdout``) written by ``split``."""
    if name not in ("test", "temporal_holdout"):
        raise EvaluationError(f"Evaluation reads only the test set or the temporal holdout, not '{name}'")
    path = config.paths.processed_dir / PARTITION_FILENAMES[name]
    if not path.is_file():
        raise EvaluationError(
            f"Partition '{name}' not found at '{path}'. Run `python -m fraud_detection.pipeline.split` first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %s: %d rows", path, len(df))
    return df, path


# ------------------------------------------------------------------------------ metrics


def rates(at_threshold: dict[str, Any]) -> dict[str, float]:
    """False-positive rate (FP / legitimate) and false-negative rate (FN / fraud)."""
    tp, fp, tn, fn = (at_threshold[k] for k in ("tp", "fp", "tn", "fn"))
    return {
        "fp_rate": fp / (fp + tn) if fp + tn else 0.0,
        "fn_rate": fn / (fn + tp) if fn + tp else 0.0,
    }


def r_min_check(at_threshold: dict[str, Any], r_min: float) -> dict[str, Any]:
    """Recall against the confirmed target, in fraud-case terms."""
    n_fraud = at_threshold["tp"] + at_threshold["fn"]
    required = math.ceil(round(r_min * n_fraud, 9))
    return {
        "r_min": r_min,
        "recall": at_threshold["recall"],
        "meets_r_min": at_threshold["recall"] >= r_min,
        "frauds_detected": at_threshold["tp"],
        "frauds_missed": at_threshold["fn"],
        "frauds_total": n_fraud,
        "frauds_required_for_r_min": required,
        "margin_in_fraud_cases": at_threshold["tp"] - required,
    }


def bootstrap_intervals(
    y_true: Any, y_proba: Any, threshold: float, n: int, confidence: float, seed: int
) -> dict[str, Any]:
    """Seeded percentile bootstrap CIs for PR-AUC, Recall and Precision at ``threshold``."""
    y = np.asarray(y_true)
    p = np.asarray(y_proba, dtype=float)
    pred = apply_threshold(p, threshold)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"pr_auc": [], "recall": [], "precision": []}
    skipped = 0
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        yb, pb, predb = y[idx], p[idx], pred[idx]
        positives = int(yb.sum())
        if positives == 0:
            skipped += 1
            continue
        tp = int((predb & yb).sum())
        predicted = int(predb.sum())
        samples["pr_auc"].append(float(average_precision_score(yb, pb)))
        samples["recall"].append(tp / positives)
        samples["precision"].append(tp / predicted if predicted else 0.0)
    alpha = (1 - confidence) / 2
    out: dict[str, Any] = {"n": n, "confidence": confidence, "seed": seed, "skipped_no_fraud": skipped}
    for metric, values in samples.items():
        low, high = np.quantile(values, [alpha, 1 - alpha])
        out[metric] = {"low": float(low), "high": float(high)}
    return out


def evaluate_predictions(
    y_true: pd.Series, y_proba: np.ndarray, frozen: FrozenModel, reference_threshold: float,
    evaluation: EvaluationConfig, seed: int,
) -> dict[str, Any]:
    """All documented metrics of the frozen model on one dataset."""
    metrics = compute_metrics(y_true, y_proba, frozen.threshold)
    result: dict[str, Any] = {
        **{k: metrics[k] for k in ("n_rows", "n_fraud", "prevalence", "pr_auc", "roc_auc", "pr_auc_lift")},
        "at_threshold": metrics["at_threshold"],
        "rates": rates(metrics["at_threshold"]),
        "r_min_check": r_min_check(metrics["at_threshold"], float(frozen.metadata["r_min"])),
        "at_reference_threshold": threshold_metrics(y_true, y_proba, reference_threshold),
    }
    if evaluation.bootstrap_enabled:
        result["bootstrap"] = bootstrap_intervals(
            y_true, y_proba, frozen.threshold, evaluation.bootstrap_n, evaluation.bootstrap_confidence, seed
        )
    return result


def frozen_identity(frozen: FrozenModel) -> dict[str, Any]:
    """Model identity copied verbatim from ``train_metadata.json``."""
    m = frozen.metadata
    return {
        "model_name": m["model_name"],
        "model_version": m["model_version"],
        "threshold": m["threshold"],
        "threshold_objective": m["threshold_objective"],
        "r_min": m["r_min"],
        "r_min_status": m.get("r_min_status"),
        "model_sha256": m["model_sha256"],
        "mlflow_run_id": m["mlflow_run_id"],
    }


def dataset_info(df: pd.DataFrame, path: Path, schema: SchemaConfig) -> dict[str, Any]:
    """Lineage of an evaluation partition."""
    fraud = int((df[schema.target] == schema.positive_label).sum())
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "rows": int(len(df)),
        "fraud_count": fraud,
        "prevalence": fraud / len(df),
    }


def flat_metrics(prefix: str, result: dict[str, Any]) -> dict[str, float]:
    """DOC-03 §6.2 metric names (``test_pr_auc``, ``temporal_recall``, ...)."""
    at, ref = result["at_threshold"], result["at_reference_threshold"]
    flat = {f"{prefix}_{k}": float(result[k]) for k in ("pr_auc", "roc_auc", "prevalence", "pr_auc_lift")}
    flat |= {f"{prefix}_{k}": float(at[k]) for k in ("precision", "recall", "f1", "f2", "tp", "fp", "tn", "fn")}
    flat |= {f"{prefix}_{k}": float(v) for k, v in result["rates"].items()}
    flat |= {f"{prefix}_ref050_{k}": float(ref[k]) for k in ("precision", "recall", "f1", "f2", "fp", "fn")}
    if "bootstrap" in result:
        for metric in ("pr_auc", "recall", "precision"):
            flat[f"{prefix}_{metric}_ci_low"] = result["bootstrap"][metric]["low"]
            flat[f"{prefix}_{metric}_ci_high"] = result["bootstrap"][metric]["high"]
    return flat
