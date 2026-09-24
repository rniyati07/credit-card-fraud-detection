"""Baseline training stage tests (DOC-05 M4): factories, outputs, reproducibility, MLflow."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient

from fraud_detection.config import CANDIDATE_MODELS, load_config
from fraud_detection.models.train import (
    TrainingError,
    build_estimator,
    compute_scale_pos_weight,
)
from fraud_detection.pipeline import split as split_stage
from fraud_detection.pipeline.train import (
    COMPARISON_FILENAME,
    SUMMARY_FILENAME,
    main,
    stage_outputs,
    train_baselines,
)


def write_split_partitions(tmp_path: Path, config_dict_factory, transactions_factory) -> Path:
    """Write a config under ``tmp_path`` and run the real split stage on synthetic data."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict_factory(tmp_path)), encoding="utf-8")
    config = load_config(config_path)
    config.paths.interim.parent.mkdir(parents=True)
    frame = transactions_factory(n_rows=2000, n_fraud=100, seed=5, signal=3.0)
    frame.to_csv(config.paths.interim, index=False)
    assert split_stage.main(["--config", str(config_path), "--log-level", "WARNING"]) == 0
    return config_path


@pytest.fixture(scope="module")
def env(tmp_path_factory, config_dict_factory, transactions_factory):
    """Config + split partitions for a small synthetic dataset, and one completed training run."""
    tmp_path = tmp_path_factory.mktemp("train")
    config_path = write_split_partitions(tmp_path, config_dict_factory, transactions_factory)
    config = load_config(config_path)
    assert main(["--config", str(config_path)]) == 0
    return config, config_path


def _read_outputs(config) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in stage_outputs(config)}


# ------------------------------------------------------------------------------ factories


def test_scale_pos_weight_is_negative_to_positive_ratio() -> None:
    assert compute_scale_pos_weight([0, 0, 0, 1]) == pytest.approx(3.0)
    with pytest.raises(TrainingError, match="both classes"):
        compute_scale_pos_weight([0, 0, 0])


def test_estimators_use_documented_imbalance_handling() -> None:
    y = np.array([0] * 9 + [1])
    lr = build_estimator("logistic_regression", {"class_weight": "balanced"}, 42, y)
    rf = build_estimator("random_forest", {"class_weight": "balanced_subsample"}, 42, y)
    xgb = build_estimator("xgboost", {}, 42, y)

    assert lr.get_params()["class_weight"] == "balanced" and lr.random_state == 42
    assert rf.get_params()["class_weight"] == "balanced_subsample" and rf.random_state == 42
    assert xgb.get_params()["scale_pos_weight"] == pytest.approx(9.0)
    assert xgb.get_params()["random_state"] == 42


def test_unknown_model_rejected() -> None:
    with pytest.raises(TrainingError, match="Unknown model"):
        build_estimator("svm", {}, 42, [0, 1])


def test_repository_config_models() -> None:
    config = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
    assert config.models.enabled == CANDIDATE_MODELS
    assert config.models.params["logistic_regression"]["class_weight"] == "balanced"
    assert config.models.params["random_forest"]["class_weight"] == "balanced_subsample"
    assert "scale_pos_weight" not in config.models.params["xgboost"]
    assert config.imbalance.strategy == "class_weight"
    assert config.threshold.reference_threshold == 0.5


# --------------------------------------------------------------------------------- outputs


def test_training_writes_all_outputs(env) -> None:
    config, _ = env
    assert all(p.is_file() for p in stage_outputs(config))

    summary = json.loads((config.paths.baseline_dir / SUMMARY_FILENAME).read_text())
    assert set(summary["models"]) == set(CANDIDATE_MODELS)
    assert summary["partitions_not_read"] == ["test", "temporal_holdout"]
    assert summary["n_features"] == 29 and "Time" not in summary["feature_order"]
    train_rows = len(pd.read_csv(config.paths.processed_dir / "train.csv"))
    assert summary["data"]["train"]["rows"] == train_rows
    assert summary["models"]["xgboost"]["imbalance_handling"]["scale_pos_weight"] == pytest.approx(
        (train_rows - summary["data"]["train"]["fraud_count"]) / summary["data"]["train"]["fraud_count"]
    )
    assert summary["models"]["random_forest"]["imbalance_handling"] == {"class_weight": "balanced_subsample"}


def test_metrics_files_contain_required_metrics(env) -> None:
    config, _ = env
    for name in CANDIDATE_MODELS:
        metrics = json.loads((config.paths.baseline_dir / "metrics" / f"{name}.json").read_text())
        for split in ("train", "val"):
            m = metrics[split]
            assert 0.0 <= m["pr_auc"] <= 1.0 and 0.0 <= m["roc_auc"] <= 1.0
            at = m["at_threshold"]
            assert {"precision", "recall", "f1", "f2", "confusion_matrix"} <= set(at)
            assert sum(map(sum, at["confusion_matrix"])) == m["n_rows"]
        assert "test" not in metrics


def test_validation_predictions_align_with_val_split(env) -> None:
    config, _ = env
    val = pd.read_csv(config.paths.processed_dir / "val.csv")
    for name in CANDIDATE_MODELS:
        preds = pd.read_csv(config.paths.baseline_dir / "predictions" / f"{name}_val.csv")
        assert list(preds.columns) == ["row_id", "Class", "fraud_probability"]
        assert preds["row_id"].tolist() == val["row_id"].tolist()
        assert preds["fraud_probability"].between(0, 1).all()


def test_comparison_ranked_by_val_pr_auc(env) -> None:
    config, _ = env
    comparison = json.loads((config.paths.baseline_dir / COMPARISON_FILENAME).read_text())
    scores = [comparison["models"][m]["val"]["pr_auc"] for m in comparison["ranking_by_val_pr_auc"]]
    assert scores == sorted(scores, reverse=True)
    assert comparison["primary_metric"] == "val_pr_auc"
    assert all(row["val_beats_no_skill"] for row in comparison["models"].values())


def test_training_is_reproducible(env) -> None:
    config, config_path = env
    first = _read_outputs(config)
    assert main(["--config", str(config_path)]) == 0
    assert _read_outputs(config) == first


# ---------------------------------------------------------------------------------- MLflow


def test_mlflow_runs_logged_with_lineage(env) -> None:
    config, _ = env
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    runs = client.search_runs([experiment.experiment_id])

    latest = {}
    for run in sorted(runs, key=lambda r: r.info.start_time):
        latest[run.data.tags["model"]] = run
    assert set(latest) == set(CANDIDATE_MODELS)

    for name, run in latest.items():
        tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
        assert tags["run_type"] == "baseline" and tags["stage"] == "train"
        for key in ("git_commit", "git_dirty", "data_sha256", "data_dvc_md5", "config_hash"):
            assert key in tags
        assert params["model"] == name and params["seed"] == "42"
        assert {"train_rows", "val_rows", "train_prevalence", "val_prevalence", "train_sha256"} <= set(params)
        assert {"val_pr_auc", "val_roc_auc", "val_ref050_precision", "val_ref050_recall",
                "val_ref050_f1", "val_ref050_fn", "train_pr_auc"} <= set(metrics)
        assert not any(k.startswith("test_") for k in metrics)
        artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
        assert {"metrics.json", "model_summary.json"} <= artifacts


def test_train_baselines_returns_run_ids(env) -> None:
    config, config_path = env
    result = train_baselines(config, config_path)
    assert set(result.run_ids) == set(CANDIDATE_MODELS)
    assert len(set(result.run_ids.values())) == 3


# ------------------------------------------------------------------------------------- CLI


def test_missing_partitions_fail_and_remove_stale_outputs(config_file) -> None:
    # No split stage has run for this config, so train.csv / val.csv are absent.
    stale = stage_outputs(load_config(config_file))[0]
    stale.parent.mkdir(parents=True)
    stale.write_text("stale")

    assert main(["--config", str(config_file)]) == 1
    assert not stale.exists()


def test_invalid_config_exit_code(tmp_path) -> None:
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2
