"""M6 entrypoint: validation threshold optimisation and rule-based model selection (DOC-05 M6).

Part of the documented ``train`` DVC stage (DOC-03 §4.2), run after ``pipeline.tune``:

1. Load the tuned candidates from ``paths.candidates_dir`` (M5) and the validation split.
2. Per candidate: validation probabilities, threshold analysis over the grid plus
   PR-curve points, and its operating point (recall >= r_min -> max precision; F2 fallback).
3. Select the model with the DOC-02 §12 framework (``models/select.py``).
4. Freeze: copy the selected candidate to ``models/model.joblib`` and write
   ``models/train_metadata.json`` (model identity, version, threshold, objective, r_min,
   feature order, seed, data hashes, library versions, validation metrics, final MLflow
   run id).

Only ``val.csv`` is read. ``train.csv`` is not needed (candidates are already fitted on
it), and ``test.csv`` / ``temporal_holdout.csv`` are never read.

Outputs:

* ``paths.reports_dir``: ``model_comparison.csv``, ``threshold_analysis_<model>.csv``,
  ``model_selection.md``
* ``paths.figures_dir``: ``threshold_analysis_<model>.png``, ``val_confusion_matrix_<model>.png``
* ``paths.models_dir``: ``model.joblib``, ``train_metadata.json``

Every output except ``train_metadata.json`` is byte-identical across reruns;
``train_metadata.json`` differs only in ``mlflow_run_id``, which DOC-05 requires it to hold.

MLflow: tuned-threshold metrics and threshold artifacts are appended to each M5
``candidate`` run (DOC-03 §6.1), and one ``run_type=final`` run is created (§6.2).

Usage::

    python -m fraud_detection.pipeline.select [--config config/config.yaml]

Exit codes: 0 = success, 1 = selection failed (missing inputs), 2 = invalid configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from fraud_detection.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from fraud_detection.data.split import write_json
from fraud_detection.evaluation.metrics import compute_metrics, threshold_metrics
from fraud_detection.evaluation.plots import plot_confusion_matrix, plot_threshold_curve
from fraud_detection.evaluation.threshold import ThresholdChoice, analyse_thresholds
from fraud_detection.models.select import (
    COMPLEXITY_RANK,
    CandidateSummary,
    SelectionResult,
    select_model,
)
from fraud_detection.models.train import TrainingError, model_params_for_logging, predict_proba
from fraud_detection.pipeline.train import features_target, load_partition, partition_info, partition_path, write_csv
from fraud_detection.pipeline.tune import CANDIDATE_FILENAMES
from fraud_detection.pipeline.tune import RUN_TYPE as CANDIDATE_RUN_TYPE
from fraud_detection.pipeline.tune import SUMMARY_FILENAME as TUNING_SUMMARY_FILENAME
from fraud_detection.tracking.mlflow_utils import (
    append_to_run,
    file_sha256,
    find_run_by_artifact,
    lineage_tags,
    log_to_active_run,
    setup_experiment,
    start_run,
)

logger = logging.getLogger("fraud_detection.pipeline.select")

STAGE = "train"
FINAL_RUN_TYPE = "final"
FINAL_RUN_NAME = "final"
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "train_metadata.json"
COMPARISON_FILENAME = "model_comparison.csv"
SELECTION_FILENAME = "model_selection.md"
PARTITION = "val"  # the only partition this step reads
NOT_READ = ("train", "test", "temporal_holdout")
REPORTED_METRICS = ("precision", "recall", "f1", "f2", "tp", "fp", "tn", "fn")

EXIT_OK = 0
EXIT_SELECTION_FAILED = 1
EXIT_CONFIG_ERROR = 2


class SelectionError(Exception):
    """Raised when selection inputs (candidates, validation data) are missing or invalid."""


@dataclass(frozen=True)
class CandidateEvaluation:
    """Validation results of one tuned candidate."""

    name: str
    candidate_path: Path
    candidate_sha256: str
    artifact_bytes: int
    metrics: dict[str, Any]
    table: pd.DataFrame
    choice: ThresholdChoice
    at_threshold: dict[str, Any]
    best_params: dict[str, Any] | None
    cv_best_average_precision: float | None
    estimator_params: dict[str, Any]


@dataclass(frozen=True)
class SelectResult:
    """Everything the step produced."""

    evaluations: dict[str, CandidateEvaluation]
    selection: SelectionResult
    comparison: pd.DataFrame
    selection_report: str
    train_metadata: dict[str, Any]
    candidate_run_ids: dict[str, str | None]
    final_run_id: str


# ------------------------------------------------------------------------------ paths


def threshold_csv_path(config: Config, model: str) -> Path:
    return config.paths.reports_dir / f"threshold_analysis_{model}.csv"


def threshold_plot_path(config: Config, model: str) -> Path:
    return config.paths.figures_dir / f"threshold_analysis_{model}.png"


def confusion_plot_path(config: Config, model: str) -> Path:
    return config.paths.figures_dir / f"val_confusion_matrix_{model}.png"


def stage_outputs(config: Config) -> list[Path]:
    """Every file this step writes for the enabled models."""
    per_model = [
        path
        for m in config.models.enabled
        for path in (threshold_csv_path(config, m), threshold_plot_path(config, m), confusion_plot_path(config, m))
    ]
    return [
        *per_model,
        config.paths.reports_dir / COMPARISON_FILENAME,
        config.paths.reports_dir / SELECTION_FILENAME,
        config.paths.models_dir / MODEL_FILENAME,
        config.paths.models_dir / METADATA_FILENAME,
    ]


def remove_stale_outputs(config: Config) -> None:
    """Delete outputs of a previous run so a failed run cannot leave a stale frozen model."""
    for path in stage_outputs(config):
        if path.exists():
            path.unlink()
            logger.warning("Removed stale output %s", path)


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    logger.info("Wrote %s", path)


# ------------------------------------------------------------------------------ inputs


def load_candidate(config: Config, name: str) -> tuple[Any, Path]:
    """Load a tuned candidate pipeline written by ``pipeline.tune``.

    Raises:
        SelectionError: If the file is missing, unreadable, or not a compatible classifier.
    """
    path = config.paths.candidates_dir / CANDIDATE_FILENAMES[name]
    if not path.is_file():
        raise SelectionError(
            f"Candidate '{name}' not found at '{path}'. Run `python -m fraud_detection.pipeline.tune` first."
        )
    try:
        pipeline = joblib.load(path)
    except Exception as exc:  # joblib/pickle raise many exception types for corrupt files
        raise SelectionError(f"Candidate '{name}' at '{path}' could not be loaded: {exc}") from exc
    if not hasattr(pipeline, "predict_proba"):
        raise SelectionError(f"Candidate '{name}' does not provide predict_proba")
    names_in = list(getattr(pipeline, "feature_names_in_", []))
    if names_in != list(config.schema.features):
        raise SelectionError(f"Candidate '{name}' was fitted on {names_in}, expected the configured feature order")
    return pipeline, path


def load_tuning_summary(config: Config) -> dict[str, Any]:
    """M5 tuning summary (best params, CV scores); empty if unavailable."""
    path = config.paths.tuning_dir / TUNING_SUMMARY_FILENAME
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("models", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        logger.warning("Tuning summary not available at %s; best params and CV scores omitted", path)
        return {}


def evaluate_candidate(
    config: Config, name: str, x_val: pd.DataFrame, y_val: pd.Series, tuning: dict[str, Any]
) -> CandidateEvaluation:
    """Validation probabilities, threshold analysis and operating point of one candidate."""
    pipeline, path = load_candidate(config, name)
    proba = predict_proba(pipeline, x_val)
    metrics = compute_metrics(y_val, proba, config.threshold.reference_threshold)
    table, choice = analyse_thresholds(y_val, proba, config.threshold)
    at_threshold = threshold_metrics(y_val, proba, choice.threshold)
    assert (at_threshold["tp"], at_threshold["fp"]) == (choice.metrics["tp"], choice.metrics["fp"])
    logger.info(
        "%-19s val PR-AUC %.4f | threshold %.6f (%s) -> P %.3f R %.3f F2 %.3f FP %d FN %d",
        name, metrics["pr_auc"], choice.threshold, choice.objective, at_threshold["precision"],
        at_threshold["recall"], at_threshold["f2"], at_threshold["fp"], at_threshold["fn"],
    )
    tuned = tuning.get(name, {})
    return CandidateEvaluation(
        name=name,
        candidate_path=path,
        candidate_sha256=file_sha256(path),
        artifact_bytes=path.stat().st_size,
        metrics=metrics,
        table=table,
        choice=choice,
        at_threshold=at_threshold,
        best_params=tuned.get("best_params"),
        cv_best_average_precision=tuned.get("cv_best_average_precision"),
        estimator_params=model_params_for_logging(pipeline),
    )


# ----------------------------------------------------------------------------- reports


def build_comparison(evaluations: dict[str, CandidateEvaluation], selection: SelectionResult) -> pd.DataFrame:
    """One row per candidate: PR-AUC, operating point at its tuned threshold, 0.50 reference."""
    order = selection.ranking + sorted(set(evaluations) - set(selection.ranking))
    rows = []
    for rank, name in enumerate(order, start=1):
        e = evaluations[name]
        ref = e.metrics["at_threshold"]
        rows.append(
            {
                "model": name,
                "selection_rank": rank if name in selection.ranking else None,
                "selected": name == selection.selected,
                "eligible": name in selection.eligible,
                "val_pr_auc": e.metrics["pr_auc"],
                "val_roc_auc": e.metrics["roc_auc"],
                "val_prevalence": e.metrics["prevalence"],
                "val_pr_auc_lift": e.metrics["pr_auc_lift"],
                "cv_best_average_precision": e.cv_best_average_precision,
                "reached_r_min": e.choice.reached_r_min,
                "threshold": e.choice.threshold,
                "threshold_objective": e.choice.objective,
                "threshold_at_grid_edge": e.choice.at_grid_edge,
                **{k: e.at_threshold[k] for k in REPORTED_METRICS},
                **{f"ref050_{k}": ref[k] for k in REPORTED_METRICS},
                "complexity_rank": COMPLEXITY_RANK.get(name),
                "artifact_bytes": e.artifact_bytes,
            }
        )
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_selection_report(
    config: Config, evaluations: dict[str, CandidateEvaluation], selection: SelectionResult, model_version: str
) -> str:
    """``model_selection.md``: every candidate's numbers and the rule-based choice (deterministic)."""
    t, s = config.threshold, evaluations[selection.selected]
    n_fraud = s.metrics["n_fraud"]
    lines = [
        "# Model Selection and Validation Threshold",
        "",
        "> Validation data only. The test split and the temporal holdout were not read. "
        "Selection is rule-based (DOC-02 §12); the threshold is chosen per DOC-02 §13. "
        "Regenerate with `python -m fraud_detection.pipeline.select`.",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| `r_min` | {t.r_min} "
        + ("(confirmed; fixed for the rest of the project, DOC-05 §23) |" if t.r_min_confirmed
           else "**(provisional: must be confirmed or revised before any test evaluation)** |"),
        f"| Threshold grid | {t.grid_start}–{t.grid_stop}, step {t.grid_step} |",
        f"| PR-curve thresholds included | {'yes' if t.include_pr_curve_points else 'no'} |",
        f"| Fallback objective | {t.fallback} |",
        f"| PR-AUC tie tolerance | {config.selection.pr_auc_tie_tolerance} (provisional) |",
        f"| Reference threshold | {t.reference_threshold} (reported only) |",
        f"| Validation rows / frauds | {s.metrics['n_rows']:,} / {n_fraud} (prevalence {s.metrics['prevalence']:.5f}) |",
        "",
        "## Candidates (validation)",
        "",
        "| Model | PR-AUC | ROC-AUC | Lift | Reaches r_min | Threshold | Objective | Precision | Recall | F1 | F2 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in selection.ranking + sorted(set(evaluations) - set(selection.ranking)):
        e, a = evaluations[name], evaluations[name].at_threshold
        marker = " **(selected)**" if name == selection.selected else ""
        lines.append(
            f"| {name}{marker} | {e.metrics['pr_auc']:.4f} | {e.metrics['roc_auc']:.4f} | "
            f"{e.metrics['pr_auc_lift']:.0f}x | {'yes' if e.choice.reached_r_min else 'no'} | "
            f"{e.choice.threshold:.6f} | {e.choice.objective} | {a['precision']:.4f} | {a['recall']:.4f} | "
            f"{a['f1']:.4f} | {a['f2']:.4f} | {a['tp']} | {a['fp']} | {a['fn']} |"
        )
    lines += [
        "",
        "Reference (threshold 0.50, not used for selection):",
        "",
        "| Model | Precision | Recall | F1 | F2 | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in selection.ranking + sorted(set(evaluations) - set(selection.ranking)):
        r = evaluations[name].metrics["at_threshold"]
        lines.append(f"| {name} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['f2']:.4f} | {r['fp']} | {r['fn']} |")

    lines += ["", "## Selection framework (DOC-02 §12)", ""]
    lines += [f"- **{step['step']}**: {step['outcome']}" for step in selection.steps]

    a, lr = s.at_threshold, evaluations.get("logistic_regression")
    grid_neighbours = s.table[s.table["source"] != "pr_curve"]
    below = grid_neighbours[grid_neighbours["threshold"] < s.choice.threshold].tail(1)
    above = grid_neighbours[grid_neighbours["threshold"] > s.choice.threshold].head(1)
    lines += [
        "",
        "## Decision",
        "",
        f"- **Selected model:** `{selection.selected}` (version `{model_version}`), frozen as `models/model.joblib`.",
        f"- **Threshold:** {s.choice.threshold:.6f} ({s.choice.objective}"
        + (", at the grid edge: reported as-is" if s.choice.at_grid_edge else "") + ").",
        f"- **At this threshold:** detects {a['tp']} of {n_fraud} validation frauds (recall {_pct(a['recall'])}), "
        f"misses {a['fn']}, flags {a['fp']} legitimate transaction{'' if a['fp'] == 1 else 's'} "
        f"(precision {_pct(a['precision'])}).",
        f"- **Granularity:** one validation fraud = {100 / n_fraud:.1f} pp of recall; small differences are noisy.",
    ]
    if lr is not None:
        sc06 = s.metrics["pr_auc"] >= lr.metrics["pr_auc"] and s.metrics["pr_auc"] > s.metrics["prevalence"]
        lines.append(
            f"- **DOC-01 SC-06:** selected PR-AUC {s.metrics['pr_auc']:.4f} vs LR {lr.metrics['pr_auc']:.4f} and "
            f"no-skill {s.metrics['prevalence']:.5f}: {'pass' if sc06 else 'FAIL'}."
        )
    lines += ["", "Neighbouring grid thresholds of the selected model (stability check, DOC-02 §13):", "",
              "| Threshold | Precision | Recall | FP | FN |", "|---|---|---|---|---|"]
    for _, row in pd.concat([below, s.table[s.table["threshold"] == s.choice.threshold], above]).iterrows():
        lines.append(f"| {row['threshold']:.6f} | {row['precision']:.4f} | {row['recall']:.4f} | {int(row['fp'])} | {int(row['fn'])} |")
    if t.r_min_confirmed:
        lines += [
            "",
            "## r_min status",
            "",
            f"`r_min = {t.r_min}` is confirmed (DOC-02 §13 / TD-12 resolved, reason logged in DOC-05 §23) and fixed "
            "for the rest of the project. The model, threshold and `r_min` must not change after the test "
            "evaluation in M7.",
            "",
        ]
    else:
        lines += [
            "",
            "## Open item",
            "",
            f"`r_min = {t.r_min}` is provisional (DOC-02 §13, TD-12). DOC-05 M6 requires it to be confirmed or "
            "revised, with the reason logged in DOC-05 §23, **before** the test set is evaluated in M7. "
            "The threshold, model and `r_min` must not change after M7.",
            "",
        ]
    return "\n".join(lines)


def library_versions() -> dict[str, str]:
    """Versions the frozen artifact depends on (DOC-02 §15; checked at serving startup, DOC-04 §4)."""
    return {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def build_train_metadata(
    config: Config,
    selected: CandidateEvaluation,
    selection: SelectionResult,
    tags: dict[str, str],
    data: dict[str, Any],
    model_version: str,
    model_path: Path,
    run_id: str,
    candidate_run_ids: dict[str, str | None],
) -> dict[str, Any]:
    """``train_metadata.json``: the frozen identity, threshold and lineage (DOC-05 M6, DOC-03 §4.2)."""
    ref = selected.metrics["at_threshold"]
    return {
        "model_name": selected.name,
        "model_version": model_version,
        "threshold": selected.choice.threshold,
        "threshold_objective": selected.choice.objective,
        "r_min": config.threshold.r_min,
        "r_min_status": "confirmed" if config.threshold.r_min_confirmed else "provisional",
        "reached_r_min": selected.choice.reached_r_min,
        "reference_threshold": config.threshold.reference_threshold,
        "feature_order": list(config.schema.features),
        "ordering_column": config.schema.ordering_column,
        "temporal_holdout_fraction": config.temporal.holdout_fraction,
        "target": config.schema.target,
        "positive_label": config.schema.positive_label,
        "random_seed": config.project.seed,
        "data_hash": tags["data_sha256"],
        "data_dvc_md5": tags["data_dvc_md5"],
        "config_hash": tags["config_hash"],
        "git_commit": tags["git_commit"],
        "git_dirty": tags["git_dirty"],
        "python_version": platform.python_version(),
        "library_versions": library_versions(),
        "data": data,
        "model_file": model_path.as_posix(),
        "model_sha256": file_sha256(model_path),
        "candidate_file": selected.candidate_path.as_posix(),
        "selected_hyperparameters": selected.best_params,
        "estimator_params": selected.estimator_params,
        "validation_metrics": {
            "pr_auc": selected.metrics["pr_auc"],
            "roc_auc": selected.metrics["roc_auc"],
            "prevalence": selected.metrics["prevalence"],
            "pr_auc_lift": selected.metrics["pr_auc_lift"],
            "n_rows": selected.metrics["n_rows"],
            "n_fraud": selected.metrics["n_fraud"],
            "at_threshold": selected.at_threshold,
            "at_reference_threshold": ref,
        },
        "selection": {
            "framework": "DOC-02 §12",
            "eligible": selection.eligible,
            "ranking_by_val_pr_auc": selection.ranking,
            "tie_group": selection.tie_group,
            "sanity_passed": selection.sanity_passed,
            "baseline_fallback": selection.baseline_fallback,
        },
        "partitions_not_read": list(NOT_READ),
        "mlflow_run_id": run_id,
        "candidate_run_ids": candidate_run_ids,
    }


def _tuned_metrics(prefix: str, e: CandidateEvaluation) -> dict[str, float]:
    """DOC-03 §6.1 tuned-threshold metric names: ``val_precision`` ... ``val_threshold``."""
    out = {f"{prefix}_{k}": float(e.at_threshold[k]) for k in ("precision", "recall", "f1", "f2", "fp", "fn", "tp", "tn")}
    out[f"{prefix}_threshold"] = e.choice.threshold
    return out


# ------------------------------------------------------------------------------- stage


def select_and_freeze(config: Config, config_path: str | Path) -> SelectResult:
    """Evaluate candidates on validation, select, freeze, write outputs and log to MLflow.

    Raises:
        SelectionError / TrainingError: If candidates or the validation split are missing/invalid.
    """
    tags = lineage_tags(
        config_path=config_path,
        raw_path=config.paths.raw,
        validation_report_path=config.paths.validation_report,
        stage=STAGE,
        run_type=FINAL_RUN_TYPE,
        output_paths=(config.paths.reports_dir, config.paths.models_dir),
    )
    val = load_partition(config, PARTITION)
    x_val, y_val = features_target(val, PARTITION, config.schema)
    tuning = load_tuning_summary(config)
    evaluations = {
        name: evaluate_candidate(config, name, x_val, y_val, tuning) for name in config.models.enabled
    }
    selection = select_model(
        {
            name: CandidateSummary(
                name=name,
                val_pr_auc=e.metrics["pr_auc"],
                reached_r_min=e.choice.reached_r_min,
                objective=e.choice.objective,
                threshold=e.choice.threshold,
                precision=e.at_threshold["precision"],
                recall=e.at_threshold["recall"],
                f2=e.at_threshold["f2"],
                fp=e.at_threshold["fp"],
                artifact_bytes=e.artifact_bytes,
            )
            for name, e in evaluations.items()
        },
        config.selection.pr_auc_tie_tolerance,
    )
    for step in selection.steps:
        logger.info("%s: %s", step["step"], step["outcome"])
    selected = evaluations[selection.selected]

    # Deterministic reports and figures.
    for name, e in evaluations.items():
        write_csv(e.table, threshold_csv_path(config, name))
        plot_threshold_curve(e.table, e.choice, name, threshold_plot_path(config, name))
        plot_confusion_matrix(
            e.at_threshold["confusion_matrix"],
            f"{name}: validation @ threshold {e.choice.threshold:.4f}",
            confusion_plot_path(config, name),
        )
    comparison = build_comparison(evaluations, selection)
    write_csv(comparison, config.paths.reports_dir / COMPARISON_FILENAME)

    short_sha = tags["git_commit"][:7] if tags["git_commit"] != "unknown" else "unknown"
    model_version = f"{config.project.model_version}+{short_sha}"
    report = render_selection_report(config, evaluations, selection, model_version)
    _write_text(report, config.paths.reports_dir / SELECTION_FILENAME)

    # Freeze: the selected candidate, already refitted on the full train split in M5.
    model_path = config.paths.models_dir / MODEL_FILENAME
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_name(model_path.name + ".tmp")
    shutil.copyfile(selected.candidate_path, tmp)
    os.replace(tmp, model_path)
    logger.info("Frozen %s -> %s", selected.candidate_path, model_path)

    # MLflow: append tuned-threshold results to the M5 candidate runs, then the final run.
    setup_experiment(config.mlflow)
    candidate_run_ids: dict[str, str | None] = {}
    for name, e in evaluations.items():
        run_id = find_run_by_artifact(
            experiment_name=config.mlflow.experiment_name,
            run_type=CANDIDATE_RUN_TYPE,
            model=name,
            artifact_name=e.candidate_path.name,
            sha256=e.candidate_sha256,
        )
        candidate_run_ids[name] = run_id
        if run_id is None:
            logger.warning("No MLflow candidate run found for %s (%s); tuned metrics not appended", name, e.candidate_path)
            continue
        append_to_run(
            run_id,
            metrics=_tuned_metrics("val", e),
            tags={"threshold_objective": e.choice.objective, "selected": str(name == selection.selected).lower()},
            json_artifacts={"val_confusion_matrix.json": {"threshold": e.choice.threshold,
                                                         "confusion_matrix": e.at_threshold["confusion_matrix"]}},
            file_artifacts=[threshold_csv_path(config, name), threshold_plot_path(config, name),
                            confusion_plot_path(config, name)],
        )

    data = {PARTITION: partition_info(val, partition_path(config, PARTITION), config.schema.target, config.schema.positive_label)}
    metadata_path = config.paths.models_dir / METADATA_FILENAME
    with start_run(run_name=FINAL_RUN_NAME, tags=tags | {"model": selected.name}) as run:
        metadata = build_train_metadata(
            config, selected, selection, tags, data, model_version, model_path, run.info.run_id, candidate_run_ids
        )
        write_json(metadata, metadata_path)
        artifacts = [
            model_path, metadata_path,
            config.paths.reports_dir / COMPARISON_FILENAME, config.paths.reports_dir / SELECTION_FILENAME,
            threshold_csv_path(config, selected.name), threshold_plot_path(config, selected.name),
            confusion_plot_path(config, selected.name), Path(config_path),
        ]
        artifacts += [p for p in (config.paths.validation_report, Path("requirements.txt")) if p.is_file()]
        log_to_active_run(
            params={
                "selected_model": selected.name,
                "threshold": selected.choice.threshold,
                "threshold_objective": selected.choice.objective,
                "r_min": config.threshold.r_min,
                "r_min_status": "confirmed" if config.threshold.r_min_confirmed else "provisional",
                "seed": config.project.seed,
                "model_version": model_version,
                "pr_auc_tie_tolerance": config.selection.pr_auc_tie_tolerance,
                "candidate_run_ids": json.dumps(candidate_run_ids, sort_keys=True),
                **{f"best_{k}": v for k, v in (selected.best_params or {}).items()},
            },
            metrics={
                "val_pr_auc": selected.metrics["pr_auc"],
                "val_roc_auc": selected.metrics["roc_auc"],
                "val_prevalence": selected.metrics["prevalence"],
                "val_pr_auc_lift": selected.metrics["pr_auc_lift"],
                **_tuned_metrics("val", selected),
                **{f"val_ref050_{k}": float(selected.metrics["at_threshold"][k]) for k in REPORTED_METRICS},
            },
            file_artifacts=artifacts,
        )
        final_run_id = run.info.run_id
    logger.info("MLflow final run %s logged for %s", final_run_id, selected.name)

    return SelectResult(evaluations, selection, comparison, report, metadata, candidate_run_ids, final_run_id)


def run_select_stage(config: Config, config_path: str | Path) -> SelectResult:
    """Run the step; remove stale outputs on failure so no stale frozen model survives."""
    try:
        return select_and_freeze(config, config_path)
    except (SelectionError, TrainingError):
        remove_stale_outputs(config)
        raise


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise validation thresholds, select the model (DOC-02 §12) and freeze it."
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
    for noisy in ("alembic", "mlflow", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLogger().level))

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("Invalid configuration: %s", exc)
        return EXIT_CONFIG_ERROR

    try:
        result = run_select_stage(config, args.config)
    except (SelectionError, TrainingError) as exc:
        logger.error("Selection FAILED: %s", exc)
        return EXIT_SELECTION_FAILED

    m = result.train_metadata
    logger.info(
        "Selected %s (version %s) with threshold %.6f (%s); val precision %.3f, recall %.3f. r_min %.2f (%s).",
        m["model_name"], m["model_version"], m["threshold"], m["threshold_objective"],
        m["validation_metrics"]["at_threshold"]["precision"], m["validation_metrics"]["at_threshold"]["recall"],
        m["r_min"], m["r_min_status"],
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
