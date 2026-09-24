"""M7 tests: frozen-artifact loading, test and temporal evaluation, model_config, MLflow, guarantees."""

from __future__ import annotations

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
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_detection.config import load_config
from fraud_detection.evaluation import evaluate as evaluate_module
from fraud_detection.evaluation.evaluate import (
    EvaluationError,
    bootstrap_intervals,
    flat_metrics,
    load_frozen_model,
    load_train_metadata,
    r_min_check,
    rates,
)
from fraud_detection.evaluation.evaluate_temporal import (
    INCONCLUSIVE,
    NO_DECREASE,
    SHIFT,
    SHIFT_WORDING,
    comparison_rows,
    interpret,
)
from fraud_detection.evaluation.metrics import compute_metrics
from fraud_detection.features.preprocess import split_features_target
from fraud_detection.pipeline import evaluate as evaluate_stage
from fraud_detection.pipeline import evaluate_temporal as temporal_stage
from fraud_detection.pipeline import select as select_stage
from fraud_detection.pipeline import split as split_stage
from fraud_detection.pipeline import tune as tune_stage

DOC02_MODEL_CONFIG_KEYS = {
    "model_name", "model_version", "threshold", "threshold_objective", "feature_order", "ordering_column",
    "temporal_holdout_fraction", "target", "positive_label", "trained_at", "random_seed", "data_hash",
    "library_versions", "validation_metrics", "test_metrics", "temporal_holdout_metrics",
}
DOC03_LINEAGE_KEYS = {"git_commit", "mlflow_run_id", "data_dvc_md5", "config_hash", "python_version"}


def _setup(tmp_path: Path, config_dict_factory, transactions_factory) -> Path:
    """Config + split + tune + select (frozen model) on synthetic data; no evaluation yet."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict_factory(tmp_path)), encoding="utf-8")
    config = load_config(config_path)
    config.paths.interim.parent.mkdir(parents=True)
    transactions_factory(n_rows=3000, n_fraud=150, seed=5, signal=3.0).to_csv(config.paths.interim, index=False)
    for stage in (split_stage, tune_stage, select_stage):
        assert stage.main(["--config", str(config_path), "--log-level", "ERROR"]) == 0
    return config_path


def _evaluate_all(config_path: Path) -> None:
    for stage in (evaluate_stage, temporal_stage):
        assert stage.main(["--config", str(config_path), "--log-level", "ERROR"]) == 0


@pytest.fixture(scope="module")
def env(tmp_path_factory, config_dict_factory, transactions_factory):
    config_path = _setup(tmp_path_factory.mktemp("evaluate"), config_dict_factory, transactions_factory)
    _evaluate_all(config_path)
    return load_config(config_path), config_path


@pytest.fixture
def fresh(tmp_path, config_dict_factory, transactions_factory):
    """A frozen model that has not been evaluated yet."""
    config_path = _setup(tmp_path, config_dict_factory, transactions_factory)
    return load_config(config_path), config_path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _outputs(config) -> list[Path]:
    return evaluate_stage.stage_outputs(config) + temporal_stage.stage_outputs(config)


# ------------------------------------------------------------------------ unit: metrics


def test_r_min_check_in_fraud_cases() -> None:
    at = {"tp": 51, "fn": 12, "recall": 51 / 63}
    check = r_min_check(at, 0.8)
    assert check["frauds_required_for_r_min"] == 51  # ceil(50.4)
    assert check["margin_in_fraud_cases"] == 0 and check["meets_r_min"]
    below = r_min_check({"tp": 39, "fn": 13, "recall": 0.75}, 0.8)
    assert below["frauds_required_for_r_min"] == 42 and below["margin_in_fraud_cases"] == -3
    assert not below["meets_r_min"]
    assert r_min_check({"tp": 4, "fn": 1, "recall": 0.8}, 0.8)["frauds_required_for_r_min"] == 4  # exact


def test_rates_from_confusion_counts() -> None:
    assert rates({"tp": 8, "fp": 2, "tn": 98, "fn": 2}) == {"fp_rate": 0.02, "fn_rate": 0.2}


def test_bootstrap_is_seeded_and_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(60, int), np.zeros(2000, int)]
    p = np.clip(np.r_[rng.normal(0.8, 0.15, 60), rng.normal(0.1, 0.1, 2000)], 0, 1)
    a = bootstrap_intervals(y, p, 0.5, 300, 0.95, seed=42)
    assert a == bootstrap_intervals(y, p, 0.5, 300, 0.95, seed=42)
    assert a != bootstrap_intervals(y, p, 0.5, 300, 0.95, seed=7)
    point = compute_metrics(y, p, 0.5)
    assert a["pr_auc"]["low"] <= point["pr_auc"] <= a["pr_auc"]["high"]
    assert a["recall"]["low"] <= point["at_threshold"]["recall"] <= a["recall"]["high"]


def _result(pr_auc, recall, precision, n_fraud=50, ci=None, fp=2, tn=1000):
    tp = round(recall * n_fraud)
    result = {
        "pr_auc": pr_auc, "roc_auc": 0.95, "prevalence": n_fraud / (n_fraud + tn + fp), "pr_auc_lift": 500.0,
        "at_threshold": {"precision": precision, "recall": recall, "f1": 0.8, "f2": 0.8,
                         "tp": tp, "fp": fp, "tn": tn, "fn": n_fraud - tp},
        "rates": {"fp_rate": 0.001, "fn_rate": 1 - recall},
        "r_min_check": {"frauds_total": n_fraud},
    }
    if ci is not None:
        result["bootstrap"] = {m: {"low": v - ci, "high": v + ci} for m, v in
                               (("pr_auc", pr_auc), ("recall", recall), ("precision", precision))}
    return result


def test_comparison_rows_follow_doc02_table() -> None:
    rows = {r["metric"]: r for r in comparison_rows(_result(0.7, 0.75, 0.9, ci=0.05), _result(0.8, 0.8, 0.95, ci=0.05))}
    assert rows["pr_auc"]["abs_delta"] == pytest.approx(-0.1)
    assert rows["pr_auc"]["rel_delta"] == pytest.approx(-0.125)
    for count in ("tp", "fp", "tn", "fn"):
        assert rows[count]["abs_delta"] is None and rows[count]["rel_delta"] is None
    assert rows["fp_rate"]["abs_delta"] is not None and rows["fp_rate"]["rel_delta"] is None
    assert rows["prevalence"]["abs_delta"] is None
    assert rows["pr_auc_lift"]["abs_delta"] == 0 and rows["pr_auc_lift"]["rel_delta"] is None
    assert rows["pr_auc"]["cis_overlap"] is True


def test_relative_delta_omitted_when_test_value_is_zero() -> None:
    rows = {r["metric"]: r for r in comparison_rows(_result(0.7, 0.8, 0.9), _result(0.7, 0.8, 0.0))}
    assert rows["precision"]["rel_delta"] is None and rows["precision"]["abs_delta"] == pytest.approx(0.9)


def test_interpretation_rules() -> None:
    test = _result(0.80, 0.80, 0.95, ci=0.02)
    better = interpret(_result(0.85, 0.82, 0.96, ci=0.02), comparison_rows(_result(0.85, 0.82, 0.96, ci=0.02), test), 2)
    assert all(v["verdict"] == NO_DECREASE for v in better["verdicts"].values())

    noisy = _result(0.75, 0.70, 0.90, ci=0.2)
    wide = interpret(noisy, comparison_rows(noisy, _result(0.80, 0.80, 0.95, ci=0.2)), 2)
    assert all(v["verdict"] == INCONCLUSIVE for v in wide["verdicts"].values())
    assert "inconclusive" in wide["overall"] and wide["metrics_decreased"] == ["pr_auc", "recall", "precision"]

    far = _result(0.50, 0.40, 0.60, ci=0.01)
    shift = interpret(far, comparison_rows(far, test), 2)
    assert all(v["verdict"] == SHIFT for v in shift["verdicts"].values())
    assert SHIFT_WORDING in shift["overall"]


def test_small_recall_change_is_inconclusive_in_fraud_cases() -> None:
    # 1 of 50 frauds fewer (-0.02 recall) with tight, non-overlapping CIs: still inconclusive.
    temporal, test = _result(0.80, 0.78, 0.95, ci=0.001), _result(0.80, 0.80, 0.95, ci=0.001)
    verdict = interpret(temporal, comparison_rows(temporal, test), 2)["verdicts"]["recall"]
    assert verdict["verdict"] == INCONCLUSIVE and "within 2 fraud cases" in verdict["reasons"]


def test_flat_metric_names_follow_doc03() -> None:
    names = set(flat_metrics("test", _result(0.8, 0.8, 0.9, ci=0.01) | {
        "at_reference_threshold": {k: 0.5 for k in ("precision", "recall", "f1", "f2", "fp", "fn")}}))
    assert {"test_pr_auc", "test_roc_auc", "test_precision", "test_recall", "test_f1", "test_f2", "test_fp",
            "test_fn", "test_prevalence", "test_pr_auc_lift", "test_ref050_precision", "test_pr_auc_ci_low"} <= names


# --------------------------------------------------------------- frozen artifact loading


def test_load_frozen_model_and_threshold(env) -> None:
    config, _ = env
    frozen = load_frozen_model(config)
    metadata = _json(config.paths.models_dir / "train_metadata.json")
    assert frozen.threshold == metadata["threshold"]
    assert list(frozen.pipeline.feature_names_in_) == list(config.schema.features)


def test_missing_metadata_keys_rejected(tmp_path) -> None:
    path = tmp_path / "train_metadata.json"
    path.write_text(json.dumps({"model_name": "x"}))
    with pytest.raises(EvaluationError, match="missing keys"):
        load_train_metadata(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m.update(model_sha256="0" * 64), "SHA-256 differs"),
        (lambda m: m.update(threshold=1.0), "Invalid frozen threshold"),
        (lambda m: m.update(feature_order=list(reversed(m["feature_order"]))), "feature_order"),
        (lambda m: m["library_versions"].update({"xgboost": "0.0.1"}), "Library version mismatch"),
    ],
)
def test_altered_frozen_artifacts_are_rejected(fresh, mutate, message) -> None:
    config, _ = fresh
    path = config.paths.models_dir / "train_metadata.json"
    metadata = _json(path)
    mutate(metadata)
    path.write_text(json.dumps(metadata))
    with pytest.raises(EvaluationError, match=message):
        load_frozen_model(config)


def test_altered_model_fails_cli_and_leaves_no_outputs(fresh) -> None:
    config, config_path = fresh
    stale = config.paths.test_metrics
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale")
    model = config.paths.models_dir / "model.joblib"
    model.write_bytes(model.read_bytes() + b"tampered")
    assert evaluate_stage.main(["--config", str(config_path), "--log-level", "ERROR"]) == 1
    assert not stale.exists()


# ---------------------------------------------------------------------- test evaluation


def test_test_metrics_match_recomputation(env) -> None:
    config, _ = env
    report = _json(config.paths.test_metrics)
    metadata = _json(config.paths.models_dir / "train_metadata.json")
    test = pd.read_csv(config.paths.processed_dir / "test.csv")
    x, y = split_features_target(test, config.schema)
    proba = joblib.load(config.paths.models_dir / "model.joblib").predict_proba(x)[:, 1]
    expected = compute_metrics(y, proba, metadata["threshold"])

    assert report["threshold"] == metadata["threshold"]  # verbatim, never changed
    assert report["model_version"] == metadata["model_version"]
    assert report["pr_auc"] == pytest.approx(expected["pr_auc"])
    assert report["roc_auc"] == pytest.approx(expected["roc_auc"])
    for key in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "f2"):
        assert report["at_threshold"][key] == pytest.approx(expected["at_threshold"][key])
    cm = report["at_threshold"]["confusion_matrix"]
    assert cm == [[report["at_threshold"]["tn"], report["at_threshold"]["fp"]],
                  [report["at_threshold"]["fn"], report["at_threshold"]["tp"]]]
    assert sum(map(sum, cm)) == report["dataset"]["rows"] == len(test)
    assert report["at_reference_threshold"]["threshold"] == 0.5
    assert report["partitions_read"] == ["test"] and "bootstrap" in report


def test_test_figures_written(env) -> None:
    config, _ = env
    for path in evaluate_stage.figure_paths(config).values():
        assert path.is_file() and path.read_bytes()[:4] == b"\x89PNG"


# ------------------------------------------------------------------ temporal evaluation


def test_temporal_outputs_and_comparison(env) -> None:
    config, _ = env
    temporal = _json(config.paths.temporal_dir / "temporal_metrics.json")
    comparison = pd.read_csv(config.paths.temporal_dir / "temporal_vs_random_comparison.csv",
                             float_precision="round_trip")
    test = _json(config.paths.test_metrics)

    assert temporal["partitions_read"] == ["temporal_holdout"]
    assert temporal["threshold"] == test["threshold"]
    assert set(comparison["metric"]) >= {"pr_auc", "roc_auc", "precision", "recall", "f1", "fp", "fn",
                                         "fp_rate", "fn_rate", "prevalence", "pr_auc_lift"}
    row = comparison.set_index("metric").loc["recall"]
    assert row["temporal_holdout"] == pytest.approx(temporal["at_threshold"]["recall"])
    assert row["random_test"] == pytest.approx(test["at_threshold"]["recall"])
    assert row["abs_delta"] == pytest.approx(row["temporal_holdout"] - row["random_test"])
    ctx = temporal["prevalence_context"]
    assert ctx["holdout_minus_pool"] == pytest.approx(ctx["holdout_prevalence"] - ctx["development_pool_prevalence"])
    assert (config.paths.temporal_dir / "temporal_confusion_matrix.png").is_file()


def test_report_wording_and_answers(env) -> None:
    config, _ = env
    report = (config.paths.temporal_dir / "temporal_robustness.md").read_text(encoding="utf-8")
    assert SHIFT_WORDING in report
    assert "production drift" not in report  # DOC-05 M7 acceptance
    for question in ("generalise to future", "How much did performance change", "acceptable according to",
                     "recall stay near the M6 target", "plateau-edge limitation"):
        assert question in report
    assert re.search(r"detected \d+ of \d+ frauds", report)
    assert "SC-10" in report


def test_model_config_assembled_verbatim(env) -> None:
    config, _ = env
    model_config = _json(config.paths.models_dir / "model_config.json")
    metadata = _json(config.paths.models_dir / "train_metadata.json")
    assert DOC02_MODEL_CONFIG_KEYS | DOC03_LINEAGE_KEYS <= set(model_config)
    for key in ("model_name", "model_version", "threshold", "threshold_objective", "feature_order",
                "mlflow_run_id", "git_commit", "data_hash", "library_versions", "validation_metrics"):
        assert model_config[key] == metadata[key]
    assert "Time" not in model_config["feature_order"] and len(model_config["feature_order"]) == 29
    assert 0 < model_config["threshold"] < 1
    candidate = MlflowClient(config.mlflow.tracking_uri).get_run(metadata["candidate_run_ids"][metadata["model_name"]])
    assert model_config["trained_at"].startswith(
        pd.Timestamp(candidate.info.start_time, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M")
    )
    assert model_config["test_metrics"]["pr_auc"] == pytest.approx(_json(config.paths.test_metrics)["pr_auc"])
    assert "delta_vs_random_test" in model_config["temporal_holdout_metrics"]


def test_temporal_requires_test_metrics(fresh) -> None:
    config, config_path = fresh
    assert temporal_stage.main(["--config", str(config_path), "--log-level", "ERROR"]) == 1
    assert not (config.paths.models_dir / "model_config.json").exists()


def test_temporal_rejects_test_metrics_of_another_model(fresh) -> None:
    config, config_path = fresh
    assert evaluate_stage.main(["--config", str(config_path), "--log-level", "ERROR"]) == 0
    report = _json(config.paths.test_metrics)
    report["threshold"] = 0.123
    config.paths.test_metrics.write_text(json.dumps(report))
    assert temporal_stage.main(["--config", str(config_path), "--log-level", "ERROR"]) == 1


# ------------------------------------------------------------------------------- MLflow


def test_mlflow_final_run_resumed_not_duplicated(env) -> None:
    config, _ = env
    metadata = _json(config.paths.models_dir / "train_metadata.json")
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    assert sum(r.data.tags.get("run_type") == "final" for r in runs) == 1

    final = client.get_run(metadata["mlflow_run_id"])
    metrics, tags = final.data.metrics, final.data.tags
    for prefix in ("test", "temporal"):
        assert {f"{prefix}_{m}" for m in ("pr_auc", "roc_auc", "precision", "recall", "f1", "f2", "fp", "fn",
                                           "prevalence", "pr_auc_lift")} <= set(metrics)
    assert {"delta_pr_auc", "delta_recall", "delta_precision", "delta_roc_auc", "delta_f1"} <= set(metrics)
    assert metrics["delta_recall"] == pytest.approx(metrics["temporal_recall"] - metrics["test_recall"])
    assert float(final.data.params["threshold"]) == metadata["threshold"]
    assert {"evaluate_git_commit", "evaluate_git_dirty", "evaluate_temporal_git_commit"} <= set(tags)
    artifacts = {a.path for a in client.list_artifacts(final.info.run_id)}
    assert {"test_metrics.json", "test_confusion_matrix.png", "test_pr_curve.png", "test_roc_curve.png",
            "temporal_metrics.json", "temporal_vs_random_comparison.csv", "temporal_confusion_matrix.png",
            "temporal_robustness.md", "model_config.json"} <= artifacts


def test_missing_final_run_fails(fresh) -> None:
    config, config_path = fresh
    raw = yaml.safe_load(config_path.read_text())
    raw["mlflow"]["tracking_uri"] = f"sqlite:///{(config_path.parent / 'empty.db').as_posix()}"
    config_path.write_text(yaml.safe_dump(raw))
    assert evaluate_stage.main(["--config", str(config_path), "--log-level", "ERROR"]) == 1


# ----------------------------------------------------------------------- guarantees


def test_each_stage_reads_only_its_partition(fresh, monkeypatch) -> None:
    config, config_path = fresh
    read: list[str] = []
    real = pd.read_csv

    def spy(path, *args, **kwargs):
        read.append(Path(path).name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    assert evaluate_stage.main(["--config", str(config_path), "--log-level", "ERROR"]) == 0
    assert read == ["test.csv"]
    read.clear()
    assert temporal_stage.main(["--config", str(config_path), "--log-level", "ERROR"]) == 0
    assert read == ["temporal_holdout.csv"]


def test_no_training_and_frozen_artifacts_unchanged(fresh, monkeypatch) -> None:
    config, config_path = fresh
    frozen_files = [config.paths.models_dir / "model.joblib", config.paths.models_dir / "train_metadata.json",
                    *config.paths.candidates_dir.iterdir()]
    before = {p: p.read_bytes() for p in frozen_files}

    def forbidden(*args, **kwargs):
        raise AssertionError("fit called during evaluation")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(XGBClassifier, "fit", forbidden)
    for cls in {type(step) for step in joblib.load(frozen_files[0]).named_steps.values()}:
        monkeypatch.setattr(cls, "fit", forbidden, raising=False)
    _evaluate_all(config_path)

    assert {p: p.read_bytes() for p in frozen_files} == before


def test_evaluation_partition_guard(env) -> None:
    config, _ = env
    for name in ("train", "val"):
        with pytest.raises(EvaluationError, match="only the test set or the temporal holdout"):
            evaluate_module.load_evaluation_partition(config, name)


# ------------------------------------------------------------------ reproducibility / CLI


def test_reruns_are_byte_identical_and_flag_repeat_evaluation(env, caplog) -> None:
    config, config_path = env
    before = {p: p.read_bytes() for p in _outputs(config)}
    caplog.set_level("WARNING")
    _evaluate_all(config_path)
    assert {p: p.read_bytes() for p in _outputs(config)} == before
    assert "already evaluated" in caplog.text
    iso = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
    for path in _outputs(config):
        if path.suffix in (".json", ".md", ".csv") and path.name != "model_config.json":
            assert not iso.search(path.read_text(encoding="utf-8")), path
    # model_config.json: the only timestamp is trained_at, a recorded fact rather than the run time.
    model_config = _json(config.paths.models_dir / "model_config.json")
    assert len(iso.findall(json.dumps(model_config))) == 1 and iso.match(model_config["trained_at"])


def test_invalid_config_exit_codes(tmp_path) -> None:
    for stage in (evaluate_stage, temporal_stage):
        assert stage.main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_bootstrap_can_be_disabled(fresh) -> None:
    config, config_path = fresh
    raw = yaml.safe_load(config_path.read_text())
    raw["evaluation"]["bootstrap"]["enabled"] = False
    config_path.write_text(yaml.safe_dump(raw))
    _evaluate_all(config_path)
    assert "bootstrap" not in _json(config.paths.test_metrics)
    report = (config.paths.temporal_dir / "temporal_robustness.md").read_text(encoding="utf-8")
    assert "Bootstrap" not in report.split("## Answers")[0]
