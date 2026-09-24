"""M6 stage tests: threshold artifacts, rule-based selection, freezing, MLflow, leakage, CLI."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient

from fraud_detection.config import CANDIDATE_MODELS, load_config
from fraud_detection.evaluation.metrics import threshold_metrics
from fraud_detection.evaluation.threshold import select_threshold
from fraud_detection.features.preprocess import split_features_target
from fraud_detection.models.train import build_model_pipeline
from fraud_detection.pipeline import split as split_stage
from fraud_detection.pipeline import tune as tune_stage
from fraud_detection.pipeline.select import (
    COMPARISON_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    SELECTION_FILENAME,
    SelectionError,
    load_candidate,
    main,
    stage_outputs,
    threshold_csv_path,
)

METADATA_KEYS = {
    "model_name", "model_version", "threshold", "threshold_objective", "r_min", "feature_order",
    "random_seed", "data_hash", "data_dvc_md5", "library_versions", "validation_metrics",
    "mlflow_run_id", "candidate_run_ids", "config_hash", "git_commit", "python_version",
}


def _setup(tmp_path: Path, config_dict_factory, transactions_factory) -> Path:
    """Config + split + M5 tuning on synthetic data (no M6 run yet)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict_factory(tmp_path)), encoding="utf-8")
    config = load_config(config_path)
    config.paths.interim.parent.mkdir(parents=True)
    transactions_factory(n_rows=2000, n_fraud=100, seed=5, signal=3.0).to_csv(config.paths.interim, index=False)
    for stage in (split_stage, tune_stage):
        assert stage.main(["--config", str(config_path), "--log-level", "ERROR"]) == 0
    return config_path


@pytest.fixture(scope="module")
def env(tmp_path_factory, config_dict_factory, transactions_factory):
    config_path = _setup(tmp_path_factory.mktemp("select"), config_dict_factory, transactions_factory)
    assert main(["--config", str(config_path), "--log-level", "WARNING"]) == 0
    return load_config(config_path), config_path


def _read_csv(path: Path) -> pd.DataFrame:
    # Thresholds are written with exact (shortest round-trip) floats; pandas' default fast
    # parser can merge values one ULP apart, so read them back exactly.
    return pd.read_csv(path, float_precision="round_trip")


def _metadata(config) -> dict:
    return json.loads((config.paths.models_dir / METADATA_FILENAME).read_text())


def _val_xy(config):
    return split_features_target(pd.read_csv(config.paths.processed_dir / "val.csv"), config.schema)


# ------------------------------------------------------------------------------ outputs


def test_all_outputs_written(env) -> None:
    config, _ = env
    missing = [p for p in stage_outputs(config) if not p.is_file()]
    assert not missing
    assert (config.paths.reports_dir / "threshold_analysis_xgboost.csv").is_file()
    assert (config.paths.figures_dir / "threshold_analysis_xgboost.png").is_file()


def test_threshold_tables_follow_documented_grid(env) -> None:
    config, _ = env
    x_val, y_val = _val_xy(config)
    for name in CANDIDATE_MODELS:
        table = _read_csv(threshold_csv_path(config, name))
        grid = table[table["source"].str.contains("grid")]
        assert len(grid) == 99
        assert table["threshold"].between(0, 1, inclusive="neither").all()
        assert table["threshold"].is_monotonic_increasing and table["threshold"].is_unique
        assert (table["source"].str.contains("pr_curve")).any()
        proba = load_candidate(config, name)[0].predict_proba(x_val)[:, 1]
        for t in (0.3, 0.5, 0.8):
            row = table[np.isclose(table["threshold"], t)].iloc[0]
            expected = threshold_metrics(y_val, proba, row["threshold"])
            assert (row["tp"], row["fp"], row["fn"]) == (expected["tp"], expected["fp"], expected["fn"])
            assert row["f2"] == pytest.approx(expected["f2"])


def test_comparison_and_chosen_thresholds(env) -> None:
    config, _ = env
    comparison = _read_csv(config.paths.reports_dir / COMPARISON_FILENAME)
    assert set(comparison["model"]) == set(CANDIDATE_MODELS)
    assert comparison["selected"].sum() == 1
    for row in comparison.itertuples():
        table = _read_csv(threshold_csv_path(config, row.model))
        choice = select_threshold(table, config.threshold.r_min, 0.01, 0.99)
        assert row.threshold == pytest.approx(choice.threshold)
        assert row.threshold_objective == choice.objective
        if row.threshold_objective == "recall>=r_min_max_precision":
            assert row.recall >= config.threshold.r_min
    assert comparison["val_pr_auc"].is_monotonic_decreasing  # ordered by selection rank


def test_frozen_model_is_selected_candidate(env) -> None:
    config, _ = env
    meta = _metadata(config)
    model_path = config.paths.models_dir / MODEL_FILENAME
    candidate = Path(meta["candidate_file"])
    assert model_path.read_bytes() == candidate.read_bytes()
    x_val, _ = _val_xy(config)
    np.testing.assert_array_equal(
        joblib.load(model_path).predict_proba(x_val), joblib.load(candidate).predict_proba(x_val)
    )


def test_train_metadata_complete(env) -> None:
    config, _ = env
    meta = _metadata(config)
    comparison = _read_csv(config.paths.reports_dir / COMPARISON_FILENAME).set_index("model")

    assert METADATA_KEYS <= set(meta)
    assert meta["model_name"] == comparison.index[comparison["selected"]].item()
    assert 0 < meta["threshold"] < 1
    assert meta["threshold"] == pytest.approx(comparison.loc[meta["model_name"], "threshold"])
    assert meta["threshold_objective"] in ("recall>=r_min_max_precision", "max_f2")
    assert meta["feature_order"] == list(config.schema.features) and "Time" not in meta["feature_order"]
    assert re.fullmatch(r"1\.0\.0\+([0-9a-f]{7}|unknown)", meta["model_version"])
    assert {"scikit-learn", "xgboost"} <= set(meta["library_versions"])
    assert meta["partitions_not_read"] == ["train", "test", "temporal_holdout"]
    assert set(meta["candidate_run_ids"]) == set(CANDIDATE_MODELS)
    assert meta["validation_metrics"]["at_threshold"]["recall"] == pytest.approx(
        comparison.loc[meta["model_name"], "recall"]
    )


def test_selection_report_lists_every_candidate(env) -> None:
    config, _ = env
    report = (config.paths.reports_dir / SELECTION_FILENAME).read_text(encoding="utf-8")
    comparison = _read_csv(config.paths.reports_dir / COMPARISON_FILENAME)
    for row in comparison.itertuples():
        assert f"| {row.model}" in report
        assert f"{row.val_pr_auc:.4f}" in report
    for step in ("1. Eligibility", "2. Primary ranking", "3. Operating-point check", "4. Tie-break", "5. LR-baseline sanity"):
        assert step in report
    assert "provisional" in report and "before** the test set" in report


def test_rerun_is_reproducible(env) -> None:
    config, config_path = env
    before = {p: p.read_bytes() for p in stage_outputs(config)}
    meta_before = _metadata(config)
    assert main(["--config", str(config_path), "--log-level", "WARNING"]) == 0

    metadata_path = config.paths.models_dir / METADATA_FILENAME
    for path, content in before.items():
        if path != metadata_path:
            assert path.read_bytes() == content, path
    meta_after = _metadata(config)
    assert meta_after["mlflow_run_id"] != meta_before["mlflow_run_id"]
    assert {k: v for k, v in meta_after.items() if k != "mlflow_run_id"} == {
        k: v for k, v in meta_before.items() if k != "mlflow_run_id"
    }


# ------------------------------------------------------------------------------- MLflow


def test_mlflow_final_and_candidate_runs(env) -> None:
    config, _ = env
    meta = _metadata(config)
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    client = MlflowClient()

    final = client.get_run(meta["mlflow_run_id"])
    tags, params, metrics = final.data.tags, final.data.params, final.data.metrics
    assert tags["run_type"] == "final" and tags["stage"] == "train"
    for key in ("git_commit", "git_dirty", "data_sha256", "data_dvc_md5", "config_hash"):
        assert key in tags
    assert params["selected_model"] == meta["model_name"]
    assert float(params["threshold"]) == pytest.approx(meta["threshold"])
    assert params["threshold_objective"] == meta["threshold_objective"]
    assert {"r_min", "seed", "model_version", "candidate_run_ids"} <= set(params)
    assert {"val_pr_auc", "val_roc_auc", "val_precision", "val_recall", "val_f1", "val_f2",
            "val_fp", "val_fn", "val_threshold", "val_ref050_precision"} <= set(metrics)
    assert not any(k.startswith(("test_", "temporal_")) for k in metrics)
    artifacts = {a.path for a in client.list_artifacts(final.info.run_id)}
    assert {MODEL_FILENAME, METADATA_FILENAME, COMPARISON_FILENAME, SELECTION_FILENAME, "config.yaml"} <= artifacts

    for name, run_id in meta["candidate_run_ids"].items():
        assert run_id is not None
        run = client.get_run(run_id)
        assert run.data.tags["run_type"] == "candidate" and run.data.tags["model"] == name
        assert {"val_threshold", "val_precision", "val_recall", "val_f2", "val_fp", "val_fn"} <= set(run.data.metrics)
        assert run.data.tags["threshold_objective"] == meta["threshold_objective"] or name != meta["model_name"]
        assert f"threshold_analysis_{name}.csv" in {a.path for a in client.list_artifacts(run_id)}


def test_missing_candidate_runs_do_not_block(tmp_path, config_dict_factory, transactions_factory) -> None:
    config_path = _setup(tmp_path, config_dict_factory, transactions_factory)
    raw = yaml.safe_load(config_path.read_text())
    raw["mlflow"]["tracking_uri"] = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"  # no M5 runs here
    config_path.write_text(yaml.safe_dump(raw))

    assert main(["--config", str(config_path), "--log-level", "ERROR"]) == 0
    meta = _metadata(load_config(config_path))
    assert meta["candidate_run_ids"] == {name: None for name in CANDIDATE_MODELS}


# ------------------------------------------------------------------------------ leakage


def test_reads_only_validation_split(tmp_path, config_dict_factory, transactions_factory, monkeypatch) -> None:
    config_path = _setup(tmp_path, config_dict_factory, transactions_factory)
    config = load_config(config_path)
    for name in ("train.csv", "test.csv", "temporal_holdout.csv"):
        (config.paths.processed_dir / name).unlink()
    read: list[str] = []
    real_read_csv = pd.read_csv

    def spy(path, *args, **kwargs):
        read.append(Path(path).name)
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    assert main(["--config", str(config_path), "--log-level", "ERROR"]) == 0
    assert read == ["val.csv"]


def test_candidate_with_wrong_feature_order_is_rejected(env, tmp_path) -> None:
    config, _ = env
    x_val, y_val = _val_xy(config)
    pipeline = build_model_pipeline("logistic_regression", {"max_iter": 200}, config.schema,
                                    config.preprocessing, 42, y_val)
    pipeline.fit(x_val[list(reversed(config.schema.features))], y_val)  # fitted on a different column order
    bad_dir = tmp_path / "candidates"
    bad_dir.mkdir()
    joblib.dump(pipeline, bad_dir / "lr.joblib")
    bad = dataclasses.replace(config, paths=dataclasses.replace(config.paths, candidates_dir=bad_dir))
    with pytest.raises(SelectionError, match="feature order"):
        load_candidate(bad, "logistic_regression")


# ---------------------------------------------------------------------------------- CLI


def test_missing_candidates_fail_and_remove_stale_frozen_model(config_file) -> None:
    config = load_config(config_file)
    stale = config.paths.models_dir / MODEL_FILENAME
    stale.parent.mkdir(parents=True)
    stale.write_text("stale")
    val = config.paths.processed_dir / "val.csv"
    val.parent.mkdir(parents=True)
    pd.DataFrame({"row_id": [0, 1], **{f: [0.0, 1.0] for f in config.schema.features},
                  "Time": [0.0, 1.0], "Class": [0, 1]}).to_csv(val, index=False)

    assert main(["--config", str(config_file)]) == 1
    assert not stale.exists()


def test_invalid_config_exit_code(tmp_path) -> None:
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_confirmed_r_min_is_reported(tmp_path, config_dict_factory, transactions_factory) -> None:
    config_path = _setup(tmp_path, config_dict_factory, transactions_factory)
    raw = yaml.safe_load(config_path.read_text())
    raw["threshold"]["r_min_confirmed"] = True
    config_path.write_text(yaml.safe_dump(raw))

    assert main(["--config", str(config_path), "--log-level", "ERROR"]) == 0
    config = load_config(config_path)
    report = (config.paths.reports_dir / SELECTION_FILENAME).read_text(encoding="utf-8")
    assert "is confirmed" in report and "## Open item" not in report
    assert _metadata(config)["r_min_status"] == "confirmed"
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    params = MlflowClient().get_run(_metadata(config)["mlflow_run_id"]).data.params
    assert params["r_min_status"] == "confirmed"
