"""DVC stage entrypoint: ``evaluate_temporal`` (DOC-03 §4.2, stage 5; DOC-05 M7).

Secondary temporal robustness evaluation (DOC-02 §21), run once after ``evaluate``:

* reads ``models/model.joblib``, ``models/train_metadata.json``, ``temporal_holdout.csv`` and
  ``reports/test_metrics.json`` (the primary results to compare against), plus the M3
  carve-out summary for pool prevalence. Never reads train/val/test data;
* evaluates the same frozen model and frozen threshold once; nothing is refitted or retuned;
* writes the DOC-02 §21.7 outputs to ``paths.temporal_dir``: ``temporal_metrics.json``,
  ``temporal_vs_random_comparison.csv``, ``temporal_confusion_matrix.png``,
  ``temporal_robustness.md``;
* assembles the final ``models/model_config.json`` (DOC-02 §15 + DOC-03 §7 lineage), copying
  identity, threshold and feature order verbatim from ``train_metadata.json``;
* resumes the M6 ``final`` MLflow run and appends ``temporal_*`` and ``delta_*`` metrics and
  artifacts.

All outputs are byte-identical across reruns (``trained_at`` is the recorded start time of the
MLflow run that trained the selected model, not the current time).

Usage::

    python -m fraud_detection.pipeline.evaluate_temporal [--config config/config.yaml]

Exit codes: 0 = success, 1 = evaluation failed (missing/altered inputs), 2 = invalid configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from fraud_detection.data.split import write_json
from fraud_detection.evaluation.evaluate import (
    MODEL_CONFIG_FILENAME,
    EvaluationError,
    FrozenModel,
    dataset_info,
    evaluate_predictions,
    flat_metrics,
    frozen_identity,
    load_evaluation_partition,
    load_frozen_model,
)
from fraud_detection.evaluation.evaluate_temporal import (
    FULL_DELTA_METRICS,
    RATE_METRICS,
    build_model_config,
    comparison_rows,
    interpret,
    load_split_summary,
    load_test_metrics,
    prevalence_context,
    render_report,
)
from fraud_detection.evaluation.plots import plot_confusion_matrix
from fraud_detection.models.train import TrainingError, predict_proba
from fraud_detection.pipeline.evaluate import check_final_run, evaluation_lineage
from fraud_detection.pipeline.train import features_target, write_csv
from fraud_detection.tracking.mlflow_utils import MlflowClient, MlflowException, append_to_run, setup_experiment

logger = logging.getLogger("fraud_detection.pipeline.evaluate_temporal")

STAGE = "evaluate_temporal"
PARTITION = "temporal_holdout"
METRICS_FILENAME = "temporal_metrics.json"
COMPARISON_FILENAME = "temporal_vs_random_comparison.csv"
CONFUSION_FIGURE = "temporal_confusion_matrix.png"
REPORT_FILENAME = "temporal_robustness.md"
RESULT_KEYS = (
    "n_rows", "n_fraud", "prevalence", "pr_auc", "roc_auc", "pr_auc_lift", "at_threshold", "rates",
    "r_min_check", "at_reference_threshold", "bootstrap",
)

EXIT_OK = 0
EXIT_EVALUATION_FAILED = 1
EXIT_CONFIG_ERROR = 2


@dataclass(frozen=True)
class TemporalResult:
    temporal_metrics: dict[str, Any]
    comparison: pd.DataFrame
    report: str
    model_config: dict[str, Any]
    final_run_id: str


def output_paths(config: Config) -> dict[str, Path]:
    d = config.paths.temporal_dir
    return {
        "metrics": d / METRICS_FILENAME,
        "comparison": d / COMPARISON_FILENAME,
        "confusion": d / CONFUSION_FIGURE,
        "report": d / REPORT_FILENAME,
        "model_config": config.paths.models_dir / MODEL_CONFIG_FILENAME,
    }


def stage_outputs(config: Config) -> list[Path]:
    return list(output_paths(config).values())


def remove_stale_outputs(config: Config) -> None:
    """Delete outputs of a previous run so a failed evaluation leaves no stale model_config.json."""
    for path in stage_outputs(config):
        if path.exists():
            path.unlink()
            logger.warning("Removed stale output %s", path)


def trained_at(frozen: FrozenModel) -> str | None:
    """Start time (UTC, ISO-8601) of the MLflow run that trained and refitted the selected model (M5)."""
    run_id = (frozen.metadata.get("candidate_run_ids") or {}).get(frozen.metadata["model_name"])
    if not run_id:
        logger.warning("No candidate run id in train_metadata.json; trained_at left null")
        return None
    try:
        start_ms = MlflowClient().get_run(run_id).info.start_time
    except MlflowException:
        logger.warning("Candidate run %s not found; trained_at left null", run_id)
        return None
    return datetime.fromtimestamp(start_ms / 1000, tz=UTC).isoformat(timespec="seconds")


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {k: result[k] for k in RESULT_KEYS if k in result}


def run_evaluate_temporal_stage(config: Config, config_path: str | Path) -> TemporalResult:
    """Evaluate the frozen model once on the temporal holdout; compare, report, assemble model_config."""
    paths = output_paths(config)
    try:
        lineage = evaluation_lineage(config, config_path, STAGE)
        frozen = load_frozen_model(config)
        setup_experiment(config.mlflow)
        run_id = frozen.metadata["mlflow_run_id"]
        if check_final_run(run_id, "temporal_pr_auc"):
            logger.warning(
                "The temporal holdout was already evaluated for this frozen model (run %s). Results are "
                "recomputed identically and never feed back into the model, threshold or r_min.", run_id,
            )
        test = load_test_metrics(config.paths.test_metrics, frozen)
        split_summary = load_split_summary(config.paths.temporal_split_summary)
        df, path = load_evaluation_partition(config, PARTITION)
        try:
            x, y = features_target(df, PARTITION, config.schema)
        except TrainingError as exc:
            raise EvaluationError(str(exc)) from exc
    except EvaluationError:
        remove_stale_outputs(config)
        raise

    proba = predict_proba(frozen.pipeline, x)
    temporal = evaluate_predictions(
        y, proba, frozen, config.threshold.reference_threshold, config.evaluation, config.project.seed
    )
    rows = comparison_rows(temporal, test)
    interpretation = interpret(temporal, rows, config.evaluation.inconclusive_fraud_cases)
    prevalence = prevalence_context(temporal, test, split_summary)
    at = temporal["at_threshold"]
    logger.info(
        "Temporal holdout: PR-AUC %.4f | ROC-AUC %.4f | P %.4f R %.4f F1 %.4f F2 %.4f | TP %d FP %d FN %d TN %d",
        temporal["pr_auc"], temporal["roc_auc"], at["precision"], at["recall"], at["f1"], at["f2"],
        at["tp"], at["fp"], at["fn"], at["tn"],
    )
    logger.info("Interpretation: %s", interpretation["overall"])

    deltas = {r["metric"]: r["abs_delta"] for r in rows if r["abs_delta"] is not None}
    temporal_metrics = {
        "stage": STAGE,
        "scope": (
            "Secondary temporal robustness evaluation of the frozen model and threshold (DOC-02 §21). "
            "Evaluated once; never used to retrain, retune, reselect or change the threshold."
        ),
        **frozen_identity(frozen),
        "reference_threshold": config.threshold.reference_threshold,
        "dataset": dataset_info(df, path, config.schema),
        "partitions_read": [PARTITION],
        **temporal,
        "delta_vs_random_test": deltas,
        "prevalence_context": prevalence,
        "interpretation": interpretation,
        "evaluation_lineage": lineage,
    }
    write_json(temporal_metrics, paths["metrics"])
    comparison = pd.DataFrame(rows)
    write_csv(comparison, paths["comparison"])
    plot_confusion_matrix(at["confusion_matrix"], f"Temporal holdout @ frozen threshold {frozen.threshold:.4f}",
                          paths["confusion"])
    report = render_report(frozen, temporal, test, rows, interpretation, prevalence, split_summary, config)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths["report"].with_name(paths["report"].name + ".tmp")
    tmp.write_text(report, encoding="utf-8", newline="\n")
    tmp.replace(paths["report"])

    model_config = build_model_config(
        frozen,
        test_metrics=_compact(test),
        temporal_metrics=_compact(temporal) | {"delta_vs_random_test": deltas, "prevalence_context": prevalence,
                                               "interpretation": interpretation["overall"]},
        trained_at=trained_at(frozen),
        evaluation_lineage={"evaluate": test.get("evaluation_lineage"), "evaluate_temporal": lineage},
    )
    write_json(model_config, paths["model_config"])

    delta_metrics = {f"delta_{m}": deltas[m] for m in (*FULL_DELTA_METRICS, *RATE_METRICS, "pr_auc_lift")}
    artifacts = [paths["metrics"], paths["comparison"], paths["confusion"], paths["report"],
                 paths["model_config"], frozen.model_path, Path(config_path)]
    artifacts += [p for p in (Path("requirements.txt"),) if p.is_file()]
    append_to_run(
        run_id,
        metrics=flat_metrics("temporal", temporal) | delta_metrics,
        tags={f"{STAGE}_{k}": v for k, v in lineage.items()}
        | {"temporal_interpretation": interpretation["overall"][:250]},
        file_artifacts=artifacts,
    )
    logger.info("Appended temporal_* and delta_* metrics and artifacts to MLflow final run %s", run_id)
    return TemporalResult(temporal_metrics, comparison, report, model_config, run_id)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen model once on the temporal holdout and compare with the test set."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns the process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    for noisy in ("alembic", "mlflow", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLogger().level))

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("Invalid configuration: %s", exc)
        return EXIT_CONFIG_ERROR

    try:
        result = run_evaluate_temporal_stage(config, args.config)
    except EvaluationError as exc:
        logger.error("Temporal evaluation FAILED: %s", exc)
        return EXIT_EVALUATION_FAILED

    check = result.temporal_metrics["r_min_check"]
    logger.info(
        "Temporal evaluation complete: recall %.4f (%d of %d frauds); model_config.json written; final run %s",
        check["recall"], check["frauds_detected"], check["frauds_total"], result.final_run_id,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
