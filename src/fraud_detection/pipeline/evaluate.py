"""DVC stage entrypoint: ``evaluate`` (DOC-03 §4.2, stage 4; DOC-05 M7).

One primary evaluation of the frozen model on the test set (DOC-02 §14):

* reads ``models/model.joblib``, ``models/train_metadata.json`` and ``test.csv`` only
  (never the temporal holdout, never validation or train data);
* loads the artifacts from disk and verifies them (round-trip), then predicts once;
* reports metrics at the frozen threshold and at 0.50 (reference), prevalence, PR-AUC lift,
  FP/FN rates, recall in fraud-case terms and seeded bootstrap CIs;
* resumes the M6 ``final`` MLflow run and appends ``test_*`` metrics and artifacts.

Nothing is refitted, retuned or reselected, and the threshold is never changed. Results are
reported as-is (DOC-02 §14 step 6).

Outputs: ``paths.test_metrics`` (``reports/test_metrics.json``) and, in ``paths.figures_dir``,
``test_confusion_matrix.png``, ``test_pr_curve.png``, ``test_roc_curve.png``. All are
byte-identical across reruns.

Usage::

    python -m fraud_detection.pipeline.evaluate [--config config/config.yaml]

Exit codes: 0 = success, 1 = evaluation failed (missing/altered inputs), 2 = invalid configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraud_detection.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from fraud_detection.data.split import write_json
from fraud_detection.evaluation.evaluate import (
    EvaluationError,
    dataset_info,
    evaluate_predictions,
    flat_metrics,
    frozen_identity,
    load_evaluation_partition,
    load_frozen_model,
)
from fraud_detection.evaluation.plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from fraud_detection.models.train import TrainingError, predict_proba
from fraud_detection.pipeline.train import features_target
from fraud_detection.tracking.mlflow_utils import (
    MlflowClient,
    MlflowException,
    append_to_run,
    lineage_tags,
    setup_experiment,
)

logger = logging.getLogger("fraud_detection.pipeline.evaluate")

STAGE = "evaluate"
PARTITION = "test"
CONFUSION_FIGURE = "test_confusion_matrix.png"
PR_FIGURE = "test_pr_curve.png"
ROC_FIGURE = "test_roc_curve.png"

EXIT_OK = 0
EXIT_EVALUATION_FAILED = 1
EXIT_CONFIG_ERROR = 2


@dataclass(frozen=True)
class EvaluateResult:
    test_metrics: dict[str, Any]
    final_run_id: str
    previously_evaluated: bool


def figure_paths(config: Config) -> dict[str, Path]:
    return {name: config.paths.figures_dir / name for name in (CONFUSION_FIGURE, PR_FIGURE, ROC_FIGURE)}


def stage_outputs(config: Config) -> list[Path]:
    return [config.paths.test_metrics, *figure_paths(config).values()]


def remove_stale_outputs(config: Config) -> None:
    """Delete outputs of a previous run so a failed evaluation leaves no stale results."""
    for path in stage_outputs(config):
        if path.exists():
            path.unlink()
            logger.warning("Removed stale output %s", path)


def evaluation_lineage(config: Config, config_path: str | Path, stage: str) -> dict[str, str]:
    """Code/config lineage of this evaluation step (the training lineage stays in train_metadata)."""
    tags = lineage_tags(
        config_path=config_path,
        raw_path=config.paths.raw,
        validation_report_path=config.paths.validation_report,
        stage=stage,
        run_type="final",
        output_paths=(config.paths.reports_dir, config.paths.models_dir),
    )
    return {k: tags[k] for k in ("git_commit", "git_dirty", "git_dirty_excludes", "config_hash")}


def check_final_run(run_id: str, metric: str) -> bool:
    """Ensure the M6 final run exists; return True if it already holds ``metric``.

    Raises:
        EvaluationError: If the run cannot be found in the configured MLflow store.
    """
    try:
        run = MlflowClient().get_run(run_id)
    except MlflowException as exc:
        raise EvaluationError(
            f"MLflow final run {run_id} (from train_metadata.json) not found in the tracking store; "
            "rerun `python -m fraud_detection.pipeline.select`"
        ) from exc
    if run.data.tags.get("run_type") != "final":
        raise EvaluationError(f"MLflow run {run_id} is not a run_type=final run")
    return metric in run.data.metrics


def run_evaluate_stage(config: Config, config_path: str | Path) -> EvaluateResult:
    """Evaluate the frozen model once on the test set and write all outputs."""
    try:
        lineage = evaluation_lineage(config, config_path, STAGE)
        frozen = load_frozen_model(config)
        setup_experiment(config.mlflow)
        run_id = frozen.metadata["mlflow_run_id"]
        previously = check_final_run(run_id, "test_pr_auc")
        if previously:
            logger.warning(
                "The test set was already evaluated for this frozen model (run %s). Results are recomputed "
                "identically and must not be used to change the model, threshold or r_min.", run_id,
            )
        df, path = load_evaluation_partition(config, PARTITION)
        try:
            x, y = features_target(df, PARTITION, config.schema)
        except TrainingError as exc:
            raise EvaluationError(str(exc)) from exc
    except EvaluationError:
        remove_stale_outputs(config)
        raise

    proba = predict_proba(frozen.pipeline, x)
    result = evaluate_predictions(
        y, proba, frozen, config.threshold.reference_threshold, config.evaluation, config.project.seed
    )
    at = result["at_threshold"]
    logger.info(
        "Test: PR-AUC %.4f | ROC-AUC %.4f | P %.4f R %.4f F1 %.4f F2 %.4f | TP %d FP %d FN %d TN %d",
        result["pr_auc"], result["roc_auc"], at["precision"], at["recall"], at["f1"], at["f2"],
        at["tp"], at["fp"], at["fn"], at["tn"],
    )

    test_metrics = {
        "stage": STAGE,
        "scope": (
            "Primary test evaluation of the frozen model and threshold (DOC-02 §14). Evaluated once; "
            "reported as-is; never used to retrain, retune or reselect."
        ),
        **frozen_identity(frozen),
        "reference_threshold": config.threshold.reference_threshold,
        "dataset": dataset_info(df, path, config.schema),
        "partitions_read": [PARTITION],
        **result,
        "evaluation_lineage": lineage,
    }
    write_json(test_metrics, config.paths.test_metrics)

    figures = figure_paths(config)
    plot_confusion_matrix(
        at["confusion_matrix"], f"Test @ frozen threshold {frozen.threshold:.4f}", figures[CONFUSION_FIGURE]
    )
    plot_pr_curve(y, proba, (at["recall"], at["precision"]), result["prevalence"], result["pr_auc"],
                  f"Test precision-recall ({frozen.metadata['model_name']})", figures[PR_FIGURE])
    plot_roc_curve(y, proba, (result["rates"]["fp_rate"], at["recall"]), result["roc_auc"],
                   f"Test ROC ({frozen.metadata['model_name']})", figures[ROC_FIGURE])

    append_to_run(
        run_id,
        metrics=flat_metrics("test", result),
        tags={f"{STAGE}_{k}": v for k, v in lineage.items()},
        file_artifacts=[config.paths.test_metrics, *figures.values()],
    )
    logger.info("Appended test_* metrics and artifacts to MLflow final run %s", run_id)
    return EvaluateResult(test_metrics, run_id, previously)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen model once on the test set.")
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
        result = run_evaluate_stage(config, args.config)
    except EvaluationError as exc:
        logger.error("Evaluation FAILED: %s", exc)
        return EXIT_EVALUATION_FAILED

    at = result.test_metrics["at_threshold"]
    logger.info(
        "Test evaluation complete: recall %.4f (%d of %d frauds), precision %.4f; final run %s",
        at["recall"], at["tp"], at["tp"] + at["fn"], at["precision"], result.final_run_id,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
