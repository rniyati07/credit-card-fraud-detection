"""M5 entrypoint: train-only hyperparameter tuning of the three candidates (DOC-05 M5).

Part of the documented ``train`` DVC stage (DOC-03 §4.2): hyperparameters are chosen by
stratified k-fold CV on the **train split only** (``scoring="average_precision"``), the
best parameters are refitted on the full train split, and each tuned candidate is then
scored once on validation. Validation metrics compare the tuned candidates; they never
choose hyperparameters. ``test.csv`` and ``temporal_holdout.csv`` are never read.

Outputs (deterministic for a given input, config and seed; no timestamps or run ids):

* ``paths.candidates_dir``: ``lr.joblib``, ``rf.joblib``, ``xgb.joblib`` (refitted pipelines)
* ``paths.tuning_dir``:
  ``tuning_summary.json`` (method, lineage, per-model best params, CV/train/val metrics,
  change vs. the M4 baseline), ``tuning_results.csv`` (every CV trial) and
  ``best_model.json`` (leader by validation PR-AUC, a provisional input to M6 selection)

MLflow: one ``run_type=candidate`` run per model (DOC-03 §6.1) with one nested
``run_type=tuning_trial`` run per CV trial.

Usage::

    python -m fraud_detection.pipeline.tune [--config config/config.yaml]

Exit codes: 0 = success, 1 = tuning failed (missing/invalid input), 2 = invalid configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fraud_detection.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from fraud_detection.data.split import write_json
from fraud_detection.evaluation.metrics import compute_metrics
from fraud_detection.models.train import TrainingError, model_params_for_logging, predict_proba
from fraud_detection.models.tuning import SearchResult, TuningError, run_search
from fraud_detection.pipeline.train import (
    COMPARISON_FILENAME,
    TRAINING_PARTITIONS,
    UNUSED_PARTITIONS,
    features_target,
    flat_metrics,
    load_partition,
    partition_info,
    partition_path,
    write_csv,
)
from fraud_detection.tracking.mlflow_utils import lineage_tags, log_run, setup_experiment

logger = logging.getLogger("fraud_detection.pipeline.tune")

STAGE = "train"  # DVC stage this step belongs to (DOC-03 §4.2)
RUN_TYPE = "candidate"
TRIAL_RUN_TYPE = "tuning_trial"
CANDIDATE_FILENAMES = {
    "logistic_regression": "lr.joblib",
    "random_forest": "rf.joblib",
    "xgboost": "xgb.joblib",
}
SUMMARY_FILENAME = "tuning_summary.json"
RESULTS_FILENAME = "tuning_results.csv"
BEST_MODEL_FILENAME = "best_model.json"
SUMMARY_METRICS = ("pr_auc", "roc_auc", "pr_auc_lift", "prevalence")
SUMMARY_THRESHOLD_METRICS = ("precision", "recall", "f1", "f2", "tp", "fp", "tn", "fn")

EXIT_OK = 0
EXIT_TUNING_FAILED = 1
EXIT_CONFIG_ERROR = 2


@dataclass(frozen=True)
class TuningResult:
    """Everything the step produced, including MLflow run ids (kept out of the files)."""

    tuning_summary: dict[str, Any]
    best_model: dict[str, Any]
    tuning_results: pd.DataFrame
    run_ids: dict[str, str]


# ----------------------------------------------------------------------------- helpers


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    at = metrics["at_threshold"]
    return {k: metrics[k] for k in SUMMARY_METRICS} | {k: at[k] for k in SUMMARY_THRESHOLD_METRICS}


def load_baseline_val_pr_auc(config: Config) -> dict[str, float]:
    """Validation PR-AUC of the untuned M4 baselines, if their comparison file exists."""
    path = config.paths.baseline_dir / COMPARISON_FILENAME
    try:
        comparison = json.loads(path.read_text(encoding="utf-8"))
        return {name: float(row["val"]["pr_auc"]) for name, row in comparison["models"].items()}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning("M4 baseline comparison not available at %s; improvement not assessed", path)
        return {}


def save_candidate(pipeline: Any, path: Path) -> Path:
    """Persist a refitted candidate pipeline with joblib, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(pipeline, tmp)
    os.replace(tmp, path)
    logger.info("Saved candidate %s", path)
    return path


def rank_candidates(val_pr_auc: dict[str, float]) -> list[str]:
    """Models ordered by validation PR-AUC (descending), ties broken by name."""
    return sorted(val_pr_auc, key=lambda m: (-val_pr_auc[m], m))


def build_best_model(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Leader of the tuned candidates by validation PR-AUC (validation metrics only)."""
    ranking = rank_candidates({m: row["val"]["pr_auc"] for m, row in models.items()})
    leader = ranking[0]
    runner_up = ranking[1] if len(ranking) > 1 else None
    return {
        "basis": "validation PR-AUC of the tuned candidates",
        "hyperparameters_chosen_by": "train-only stratified CV average precision (validation not used)",
        "status": (
            "Provisional leader for M6. Final selection applies the full DOC-02 §12 framework "
            "(eligibility, PR-AUC, operating point, tie-break, LR-baseline sanity) and may differ."
        ),
        "not_used": ["test", "temporal_holdout", "threshold tuning"],
        "leader": leader,
        "leader_val_pr_auc": models[leader]["val"]["pr_auc"],
        "runner_up": runner_up,
        "margin_over_runner_up": (
            models[leader]["val"]["pr_auc"] - models[runner_up]["val"]["pr_auc"] if runner_up else None
        ),
        "ranking_by_val_pr_auc": ranking,
        "best_per_model": {
            name: {
                "candidate_file": models[name]["candidate_file"],
                "best_params": models[name]["best_params"],
                "cv_best_average_precision": models[name]["cv_best_average_precision"],
                "val_pr_auc": models[name]["val"]["pr_auc"],
                "val_roc_auc": models[name]["val"]["roc_auc"],
            }
            for name in ranking
        },
    }


def stage_outputs(config: Config) -> list[Path]:
    """Every file this step writes for the enabled models."""
    tuning_dir = config.paths.tuning_dir
    return [
        *(config.paths.candidates_dir / CANDIDATE_FILENAMES[m] for m in config.models.enabled),
        tuning_dir / SUMMARY_FILENAME,
        tuning_dir / RESULTS_FILENAME,
        tuning_dir / BEST_MODEL_FILENAME,
    ]


def remove_stale_outputs(config: Config) -> None:
    """Delete outputs of a previous run so they cannot be mistaken for the current one."""
    for path in stage_outputs(config):
        if path.exists():
            path.unlink()
            logger.warning("Removed stale output %s", path)


def _trial_children(search: SearchResult, tags: dict[str, str], cv_folds: int) -> list[dict[str, Any]]:
    children = []
    for row in search.trials.sort_values("trial").itertuples(index=False):
        row_dict = row._asdict()
        params = json.loads(row_dict["params"])
        children.append(
            {
                "run_name": f"{search.model}_trial_{row_dict['trial']:02d}",
                "params": {"model": search.model, "trial": row_dict["trial"], **params},
                "metrics": {
                    "cv_rank": float(row_dict["rank"]),
                    "cv_mean_average_precision": row_dict["mean_cv_average_precision"],
                    "cv_std_average_precision": row_dict["std_cv_average_precision"],
                    **{
                        f"cv_split{k}_average_precision": row_dict[f"split{k}_cv_average_precision"]
                        for k in range(cv_folds)
                    },
                },
                "tags": tags | {"model": search.model, "run_type": TRIAL_RUN_TYPE},
            }
        )
    return children


# ------------------------------------------------------------------------------- stage


def tune_candidates(config: Config, config_path: str | Path) -> TuningResult:
    """Tune, refit and score every enabled candidate; log MLflow runs; return all results.

    Raises:
        TrainingError: If a partition is missing or invalid.
        TuningError: If a search cannot be built or run.
    """
    schema, seed, threshold = config.schema, config.project.seed, config.threshold.reference_threshold
    tuning = config.tuning
    frames = {name: load_partition(config, name) for name in TRAINING_PARTITIONS}
    (x_train, y_train), (x_val, y_val) = (features_target(frames[n], n, schema) for n in TRAINING_PARTITIONS)
    data = {
        name: partition_info(frames[name], partition_path(config, name), schema.target, schema.positive_label)
        for name in TRAINING_PARTITIONS
    }
    baseline = load_baseline_val_pr_auc(config)
    tags = lineage_tags(
        config_path=config_path,
        raw_path=config.paths.raw,
        validation_report_path=config.paths.validation_report,
        stage=STAGE,
        run_type=RUN_TYPE,
        output_paths=(config.paths.reports_dir, config.paths.models_dir),
    )
    setup_experiment(config.mlflow)

    models: dict[str, dict[str, Any]] = {}
    trials: list[pd.DataFrame] = []
    run_ids: dict[str, str] = {}
    for name in config.models.enabled:
        search = run_search(
            name, config.models.params[name], tuning.search_spaces[name], tuning,
            schema, config.preprocessing, seed, x_train, y_train,
        )
        pipeline = search.best_pipeline
        metrics = {
            "model": name,
            "train": compute_metrics(y_train, predict_proba(pipeline, x_train), threshold),
            "val": compute_metrics(y_val, predict_proba(pipeline, x_val), threshold),
        }
        candidate_path = save_candidate(pipeline, config.paths.candidates_dir / CANDIDATE_FILENAMES[name])
        estimator_params = model_params_for_logging(pipeline)

        val_pr_auc = metrics["val"]["pr_auc"]
        baseline_pr_auc = baseline.get(name)
        delta = val_pr_auc - baseline_pr_auc if baseline_pr_auc is not None else None
        models[name] = {
            "search_strategy": search.strategy,
            "n_candidates": search.n_candidates,
            "search_space": {k: list(v) for k, v in tuning.search_spaces[name].items()},
            "best_params": search.best_params,
            "cv_best_average_precision": search.cv_best_score,
            "cv_best_std": search.cv_best_std,
            "estimator_params": estimator_params,
            "train": _metric_summary(metrics["train"]),
            "val": _metric_summary(metrics["val"]),
            "val_confusion_matrix_ref": metrics["val"]["at_threshold"]["confusion_matrix"],
            "baseline_val_pr_auc": baseline_pr_auc,
            "val_pr_auc_change_vs_baseline": delta,
            "improved_over_baseline": None if delta is None else delta > 0,
            "candidate_file": candidate_path.as_posix(),
        }
        if delta is not None:
            log = logger.info if delta > 0 else logger.warning
            log(
                "%s val PR-AUC %.4f vs M4 baseline %.4f (%+.4f)%s",
                name, val_pr_auc, baseline_pr_auc, delta, "" if delta > 0 else ": tuning did not improve validation PR-AUC",
            )
        trials.append(search.trials)

        run_params = {
            "model": name,
            "seed": seed,
            "imbalance_strategy": config.imbalance.strategy,
            "n_features": len(schema.features),
            "reference_threshold": threshold,
            "cv_folds": tuning.cv_folds,
            "n_iter": tuning.n_iter,
            "scoring": tuning.scoring,
            "search_strategy": search.strategy,
            "n_candidates": search.n_candidates,
            "search_space": json.dumps(models[name]["search_space"], sort_keys=True),
            **{f"best_{k}": v for k, v in search.best_params.items()},
            **{f"{p}_{k}": data[p][k] for p in TRAINING_PARTITIONS for k in ("rows", "fraud_count", "prevalence", "sha256")},
            **estimator_params,
        }
        run_metrics = {
            "cv_best_average_precision": search.cv_best_score,
            "cv_best_std": search.cv_best_std,
            **flat_metrics("train", metrics["train"], threshold),
            **flat_metrics("val", metrics["val"], threshold),
        }
        if delta is not None:
            run_metrics["val_pr_auc_change_vs_baseline"] = delta
        run_ids[name] = log_run(
            run_name=name,
            params=run_params,
            metrics=run_metrics,
            tags=tags | {"model": name},
            json_artifacts={
                "metrics.json": metrics,
                "tuning_summary.json": models[name],
                "cv_results.json": json.loads(search.trials.to_json(orient="records")),
            },
            file_artifacts=[candidate_path],
            children=_trial_children(search, tags, tuning.cv_folds),
        )

    tuning_results = pd.concat(trials, ignore_index=True)
    tuning_summary = {
        "stage": STAGE,
        "scope": "M5 hyperparameter tuning (no threshold tuning, no final selection, no test evaluation)",
        "seed": seed,
        "method": {
            "search": "GridSearchCV if the space has <= n_iter combinations, else RandomizedSearchCV",
            "cv": f"StratifiedKFold(n_splits={tuning.cv_folds}, shuffle=True, random_state={seed})",
            "scoring": tuning.scoring,
            "n_iter": tuning.n_iter,
            "fit_data": "train split only; preprocessing refitted inside every fold",
            "refit": "best parameters refitted on the full train split",
            "validation_use": "scoring the refitted candidates only (never for choosing hyperparameters)",
        },
        "reference_threshold": threshold,
        "imbalance_strategy": config.imbalance.strategy,
        "feature_order": list(schema.features),
        "data": data,
        "partitions_not_read": list(UNUSED_PARTITIONS),
        "lineage": {k: tags[k] for k in ("data_sha256", "data_dvc_md5", "config_hash")},
        "mlflow": {
            "tracking_uri": config.mlflow.tracking_uri,
            "experiment_name": config.mlflow.experiment_name,
            "run_type": RUN_TYPE,
            "trial_run_type": TRIAL_RUN_TYPE,
        },
        "models": models,
    }
    return TuningResult(tuning_summary, build_best_model(models), tuning_results, run_ids)


def run_tune_stage(config: Config, config_path: str | Path) -> TuningResult:
    """Tune candidates and write all outputs; remove stale outputs on failure."""
    try:
        result = tune_candidates(config, config_path)
    except (TrainingError, TuningError):
        remove_stale_outputs(config)
        raise
    tuning_dir = config.paths.tuning_dir
    write_json(result.tuning_summary, tuning_dir / SUMMARY_FILENAME)
    write_json(result.best_model, tuning_dir / BEST_MODEL_FILENAME)
    write_csv(result.tuning_results, tuning_dir / RESULTS_FILENAME)
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune LR / RF / XGBoost with train-only stratified CV and track candidates in MLflow."
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
    for noisy in ("alembic", "mlflow"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLogger().level))

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("Invalid configuration: %s", exc)
        return EXIT_CONFIG_ERROR

    try:
        result = run_tune_stage(config, args.config)
    except (TrainingError, TuningError) as exc:
        logger.error("Tuning FAILED: %s", exc)
        return EXIT_TUNING_FAILED

    best = result.best_model
    logger.info(
        "Tuning complete. Val PR-AUC ranking: %s. Provisional leader: %s",
        ", ".join(f"{m} ({best['best_per_model'][m]['val_pr_auc']:.4f})" for m in best["ranking_by_val_pr_auc"]),
        best["leader"],
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
