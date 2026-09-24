"""DVC stage entrypoint: ``train`` (DOC-03 §4.2, stage 3). M4 scope: baseline training.

Trains every enabled candidate (LR, RF, XGBoost) with cost-sensitive imbalance handling
and fixed/default hyperparameters on the **train split only**, scores train and
validation, and tracks one MLflow run per model (``run_type=baseline``).

Reads only ``train.csv`` and ``val.csv``. ``test.csv`` and ``temporal_holdout.csv`` are
never opened (DOC-03 MD-06, DOC-05 M4 "Must NOT"). No threshold tuning or model
selection happens here: threshold-dependent metrics use the 0.50 reference only.

Writes to ``paths.baseline_dir`` (deterministic for a given input, config and seed):

* ``training_summary.json``: data lineage, split sizes/prevalence, resolved model params
* ``model_comparison.json``: train/val metrics per model, ordered by val PR-AUC
* ``metrics/<model>.json``: full metrics including confusion matrices
* ``predictions/<model>_val.csv``: validation probabilities (``row_id``, target, proba)

Usage::

    python -m fraud_detection.pipeline.train [--config config/config.yaml]

Exit codes: 0 = success, 1 = training failed (missing/invalid input), 2 = invalid configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.config import DEFAULT_CONFIG_PATH, Config, ConfigError, SchemaConfig, load_config
from fraud_detection.data.split import PARTITION_FILENAMES, write_json
from fraud_detection.evaluation.metrics import compute_metrics
from fraud_detection.features.preprocess import split_features_target
from fraud_detection.models.train import (
    MODEL_STEP,
    TrainingError,
    build_model_pipeline,
    fit_model_pipeline,
    model_params_for_logging,
    predict_proba,
)
from fraud_detection.tracking.mlflow_utils import file_sha256, lineage_tags, log_run, setup_experiment

logger = logging.getLogger("fraud_detection.pipeline.train")

STAGE = "train"
RUN_TYPE = "baseline"
# The only partitions this stage may read (structural leakage guard, DOC-05 §8).
TRAINING_PARTITIONS = ("train", "val")
UNUSED_PARTITIONS = ("test", "temporal_holdout")
SUMMARY_FILENAME = "training_summary.json"
COMPARISON_FILENAME = "model_comparison.json"
METRICS_DIRNAME = "metrics"
PREDICTIONS_DIRNAME = "predictions"
METRIC_KEYS = ("pr_auc", "roc_auc", "pr_auc_lift", "prevalence")
THRESHOLD_KEYS = ("precision", "recall", "f1", "f2", "tp", "fp", "tn", "fn")

EXIT_OK = 0
EXIT_TRAINING_FAILED = 1
EXIT_CONFIG_ERROR = 2


@dataclass(frozen=True)
class BaselineResult:
    """Everything the stage produced, including MLflow run ids (kept out of the files)."""

    training_summary: dict[str, Any]
    model_comparison: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    val_predictions: dict[str, pd.DataFrame]
    run_ids: dict[str, str]


# ------------------------------------------------------------------------------ inputs


def partition_path(config: Config, name: str) -> Path:
    """Path of a partition this stage is allowed to read.

    Raises:
        TrainingError: If ``name`` is not a training partition (test/holdout are off limits).
    """
    if name not in TRAINING_PARTITIONS:
        raise TrainingError(f"The train stage must not read the '{name}' partition")
    return config.paths.processed_dir / PARTITION_FILENAMES[name]


def load_partition(config: Config, name: str) -> pd.DataFrame:
    """Load the train or val partition written by the ``split`` stage."""
    path = partition_path(config, name)
    if not path.is_file():
        raise TrainingError(
            f"Partition '{name}' not found at '{path}'. Run `python -m fraud_detection.pipeline.split` first."
        )
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        raise TrainingError(f"Partition '{name}' at '{path}' could not be read: {exc}") from exc
    logger.info("Loaded %s: %d rows", path, len(df))
    return df


def features_target(df: pd.DataFrame, name: str, schema: SchemaConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Model inputs and target of a partition, which must contain both classes."""
    try:
        x, y = split_features_target(df, schema)
    except ValueError as exc:
        raise TrainingError(f"Partition '{name}': {exc}") from exc
    if set(y.unique()) != {0, 1}:
        raise TrainingError(f"Partition '{name}' must contain both classes, found {sorted(y.unique())}")
    return x, y


def partition_info(df: pd.DataFrame, path: Path, target: str, positive_label: int) -> dict[str, Any]:
    """Lineage and size of a partition: path, SHA-256, rows, frauds, prevalence."""
    fraud = int((df[target] == positive_label).sum())
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "rows": int(len(df)),
        "fraud_count": fraud,
        "prevalence": fraud / len(df),
    }


# ----------------------------------------------------------------------------- reports


def flat_metrics(prefix: str, metrics: dict[str, Any], threshold: float) -> dict[str, float]:
    """MLflow metric names per DOC-03 §6.1 (``val_pr_auc``, ``val_ref050_precision``, ...)."""
    ref = f"ref{round(threshold * 100):03d}"
    flat = {f"{prefix}_{k}": float(metrics[k]) for k in METRIC_KEYS}
    flat |= {f"{prefix}_{ref}_{k}": float(metrics["at_threshold"][k]) for k in THRESHOLD_KEYS}
    return flat


def _comparison_row(metrics: dict[str, Any]) -> dict[str, Any]:
    at = metrics["at_threshold"]
    return {k: metrics[k] for k in METRIC_KEYS} | {k: at[k] for k in THRESHOLD_KEYS}


def build_model_comparison(
    metrics: dict[str, dict[str, Any]], threshold: float
) -> dict[str, Any]:
    """Side-by-side train/val metrics, ordered by validation PR-AUC. Comparison only."""
    ranking = sorted(metrics, key=lambda m: (-metrics[m]["val"]["pr_auc"], m))
    return {
        "primary_metric": "val_pr_auc",
        "reference_threshold": threshold,
        "note": (
            "Baseline comparison on validation only (untuned models). Threshold-dependent "
            "metrics use the 0.50 reference. Model selection and threshold tuning happen in M6; "
            "the test set is evaluated once in M7."
        ),
        "ranking_by_val_pr_auc": ranking,
        "models": {
            name: {
                "val_beats_no_skill": metrics[name]["val"]["pr_auc"] > metrics[name]["val"]["prevalence"],
                "train": _comparison_row(metrics[name]["train"]),
                "val": _comparison_row(metrics[name]["val"]),
            }
            for name in ranking
        },
    }


def stage_outputs(config: Config) -> list[Path]:
    """Every file this stage writes for the enabled models."""
    base = config.paths.baseline_dir
    files = [base / SUMMARY_FILENAME, base / COMPARISON_FILENAME]
    for name in config.models.enabled:
        files += [base / METRICS_DIRNAME / f"{name}.json", base / PREDICTIONS_DIRNAME / f"{name}_val.csv"]
    return files


def remove_stale_outputs(config: Config) -> None:
    """Delete outputs of a previous run so they cannot be mistaken for the current one."""
    for path in stage_outputs(config):
        if path.exists():
            path.unlink()
            logger.warning("Removed stale output %s", path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV (no index, LF line endings) atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)
    logger.info("Wrote %s (%d rows)", path, len(df))


def write_outputs(result: BaselineResult, config: Config) -> None:
    """Write summary, comparison, per-model metrics and validation probabilities."""
    base = config.paths.baseline_dir
    write_json(result.training_summary, base / SUMMARY_FILENAME)
    write_json(result.model_comparison, base / COMPARISON_FILENAME)
    for name, metrics in result.metrics.items():
        write_json(metrics, base / METRICS_DIRNAME / f"{name}.json")
        write_csv(result.val_predictions[name], base / PREDICTIONS_DIRNAME / f"{name}_val.csv")


# ------------------------------------------------------------------------------- stage


def train_baselines(config: Config, config_path: str | Path) -> BaselineResult:
    """Fit each enabled candidate on train, score train/val, and log one MLflow run per model.

    Raises:
        TrainingError: If a partition is missing/invalid or a model cannot be trained.
    """
    schema, seed, threshold = config.schema, config.project.seed, config.threshold.reference_threshold
    frames = {name: load_partition(config, name) for name in TRAINING_PARTITIONS}
    (x_train, y_train), (x_val, y_val) = (features_target(frames[n], n, schema) for n in TRAINING_PARTITIONS)

    data = {
        name: partition_info(frames[name], partition_path(config, name), schema.target, schema.positive_label)
        for name in TRAINING_PARTITIONS
    }
    tags = lineage_tags(
        config_path=config_path,
        raw_path=config.paths.raw,
        validation_report_path=config.paths.validation_report,
        stage=STAGE,
        run_type=RUN_TYPE,
    )
    setup_experiment(config.mlflow)

    models_summary: dict[str, Any] = {}
    all_metrics: dict[str, dict[str, Any]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    run_ids: dict[str, str] = {}
    for name in config.models.enabled:
        try:
            pipeline = build_model_pipeline(
                name, config.models.params[name], schema, config.preprocessing, seed, y_train
            )
            fit_model_pipeline(pipeline, x_train, y_train)
        except ValueError as exc:
            raise TrainingError(f"Training '{name}' failed: {exc}") from exc

        p_train, p_val = predict_proba(pipeline, x_train), predict_proba(pipeline, x_val)
        metrics = {
            "model": name,
            "train": compute_metrics(y_train, p_train, threshold),
            "val": compute_metrics(y_val, p_val, threshold),
        }
        params = model_params_for_logging(pipeline)
        estimator = pipeline.named_steps[MODEL_STEP]
        models_summary[name] = {
            "estimator": f"{type(estimator).__module__}.{type(estimator).__name__}",
            "imbalance_handling": (
                {"scale_pos_weight": params["scale_pos_weight"]}
                if name == "xgboost"
                else {"class_weight": params["class_weight"]}
            ),
            "params": params,
        }
        all_metrics[name] = metrics
        predictions[name] = pd.DataFrame(
            {
                config.split.row_id_column: frames["val"][config.split.row_id_column],
                schema.target: y_val,
                "fraud_probability": p_val,
            }
        )
        logger.info(
            "%-19s val PR-AUC %.4f | ROC-AUC %.4f | @%.2f P %.3f R %.3f F1 %.3f (FP %d, FN %d)",
            name, metrics["val"]["pr_auc"], metrics["val"]["roc_auc"], threshold,
            metrics["val"]["at_threshold"]["precision"], metrics["val"]["at_threshold"]["recall"],
            metrics["val"]["at_threshold"]["f1"], metrics["val"]["at_threshold"]["fp"],
            metrics["val"]["at_threshold"]["fn"],
        )

        run_params = {
            "model": name,
            "seed": seed,
            "imbalance_strategy": config.imbalance.strategy,
            "n_features": len(schema.features),
            "reference_threshold": threshold,
            **{f"{p}_{k}": data[p][k] for p in TRAINING_PARTITIONS for k in ("rows", "fraud_count", "prevalence", "sha256")},
            **params,
        }
        run_metrics = flat_metrics("train", metrics["train"], threshold) | flat_metrics(
            "val", metrics["val"], threshold
        )
        run_ids[name] = log_run(
            run_name=f"baseline_{name}",
            params=run_params,
            metrics=run_metrics,
            tags=tags | {"model": name},
            json_artifacts={"metrics.json": metrics, "model_summary.json": models_summary[name]},
        )

    training_summary = {
        "stage": STAGE,
        "scope": "M4 baseline training (untuned; no threshold tuning, no model selection)",
        "seed": seed,
        "reference_threshold": threshold,
        "imbalance_strategy": config.imbalance.strategy,
        "target": schema.target,
        "feature_order": list(schema.features),
        "n_features": len(schema.features),
        "data": data,
        "partitions_not_read": list(UNUSED_PARTITIONS),
        "lineage": {k: tags[k] for k in ("data_sha256", "data_dvc_md5", "config_hash")},
        "mlflow": {
            "tracking_uri": config.mlflow.tracking_uri,
            "experiment_name": config.mlflow.experiment_name,
            "run_type": RUN_TYPE,
        },
        "models": models_summary,
    }
    comparison = build_model_comparison(all_metrics, threshold)
    for name, row in comparison["models"].items():
        if not row["val_beats_no_skill"]:
            logger.warning("%s validation PR-AUC does not exceed prevalence (no-skill)", name)
    return BaselineResult(training_summary, comparison, all_metrics, predictions, run_ids)


def run_train_stage(config: Config, config_path: str | Path) -> BaselineResult:
    """Train baselines and write all outputs; remove stale outputs on failure."""
    try:
        result = train_baselines(config, config_path)
    except TrainingError:
        remove_stale_outputs(config)
        raise
    write_outputs(result, config)
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline LR / RF / XGBoost models on the train split and track them in MLflow."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns the process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    for noisy in ("alembic", "mlflow"):  # DB migrations and tracking chatter
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLogger().level))

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("Invalid configuration: %s", exc)
        return EXIT_CONFIG_ERROR

    try:
        result = run_train_stage(config, args.config)
    except TrainingError as exc:
        logger.error("Training FAILED: %s", exc)
        return EXIT_TRAINING_FAILED

    ranking = result.model_comparison["ranking_by_val_pr_auc"]
    logger.info(
        "Baseline training complete: %d models; val PR-AUC ranking: %s; MLflow runs: %s",
        len(ranking),
        ", ".join(f"{m} ({result.metrics[m]['val']['pr_auc']:.4f})" for m in ranking),
        ", ".join(f"{m}={rid[:8]}" for m, rid in result.run_ids.items()),
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
