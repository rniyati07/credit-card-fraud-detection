"""Hyperparameter tuning tests (DOC-05 M5): search setup, selection, outputs, MLflow, leakage."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from fraud_detection.config import CANDIDATE_MODELS, load_config
from fraud_detection.evaluation.metrics import compute_metrics
from fraud_detection.features.preprocess import split_features_target
from fraud_detection.models import tuning as tuning_module
from fraud_detection.models.train import PREPROCESS_STEP, build_model_pipeline
from fraud_detection.models.tuning import (
    TuningError,
    build_search,
    cv_results_frame,
    run_search,
    search_space_size,
)
from fraud_detection.pipeline import split as split_stage
from fraud_detection.pipeline import train as train_stage
from fraud_detection.pipeline import tune as tune_stage
from fraud_detection.pipeline.tune import (
    BEST_MODEL_FILENAME,
    RESULTS_FILENAME,
    SUMMARY_FILENAME,
    build_best_model,
    load_baseline_val_pr_auc,
    main,
    rank_candidates,
    stage_outputs,
)


def _setup(tmp_path: Path, config_dict_factory, transactions_factory, run_baseline: bool = True) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict_factory(tmp_path)), encoding="utf-8")
    config = load_config(config_path)
    config.paths.interim.parent.mkdir(parents=True)
    transactions_factory(n_rows=2000, n_fraud=100, seed=5, signal=3.0).to_csv(config.paths.interim, index=False)
    assert split_stage.main(["--config", str(config_path), "--log-level", "WARNING"]) == 0
    if run_baseline:
        assert train_stage.main(["--config", str(config_path), "--log-level", "WARNING"]) == 0
    return config_path


@pytest.fixture(scope="module")
def env(tmp_path_factory, config_dict_factory, transactions_factory):
    """Split + M4 baseline + one completed M5 tuning run on synthetic data."""
    config_path = _setup(tmp_path_factory.mktemp("tune"), config_dict_factory, transactions_factory)
    assert main(["--config", str(config_path), "--log-level", "WARNING"]) == 0
    return load_config(config_path), config_path


@pytest.fixture
def train_xy(env):
    config, _ = env
    df = pd.read_csv(config.paths.processed_dir / "train.csv")
    return split_features_target(df, config.schema)


def _read(config, filename: str):
    path = config.paths.tuning_dir / filename
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


# ------------------------------------------------------------------------- search setup


def test_search_space_size() -> None:
    assert search_space_size({"a": (1, 2), "b": (1, 2, 3)}) == 6
    assert search_space_size({"C": (0.01, 0.1, 1, 10)}) == 4


def test_small_space_uses_exhaustive_grid(env, train_xy) -> None:
    config, _ = env
    _, y = train_xy
    pipe = build_model_pipeline("logistic_regression", config.models.params["logistic_regression"],
                                config.schema, config.preprocessing, 42, y)
    search = build_search(pipe, {"C": (0.1, 1.0)}, config.tuning, 42)

    assert isinstance(search, GridSearchCV)
    assert search.param_grid == {"model__C": [0.1, 1.0]}


def test_large_space_uses_seeded_randomized_search(env, train_xy) -> None:
    config, _ = env
    _, y = train_xy
    space = config.tuning.search_spaces["random_forest"]  # 4 combinations > n_iter=3
    pipe = build_model_pipeline("random_forest", config.models.params["random_forest"],
                                config.schema, config.preprocessing, 42, y)
    search = build_search(pipe, space, config.tuning, 42)

    assert isinstance(search, RandomizedSearchCV)
    assert search.n_iter == config.tuning.n_iter and search.random_state == 42
    assert search.scoring == "average_precision" and search.refit is True
    assert isinstance(search.cv, StratifiedKFold)
    assert (search.cv.n_splits, search.cv.shuffle, search.cv.random_state) == (3, True, 42)


def test_repository_config_matches_documented_spaces() -> None:
    config = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
    t = config.tuning
    assert (t.cv_folds, t.n_iter, t.scoring) == (3, 10, "average_precision")
    assert t.search_spaces["logistic_regression"] == {"C": (0.01, 0.1, 1, 10)}
    assert t.search_spaces["random_forest"] == {
        "n_estimators": (200, 400), "max_depth": (None, 10, 20), "min_samples_leaf": (1, 3, 5),
    }
    assert t.search_spaces["xgboost"] == {
        "n_estimators": (200, 400, 600), "max_depth": (3, 4, 6), "learning_rate": (0.03, 0.1),
        "subsample": (0.8, 1.0), "colsample_bytree": (0.8, 1.0), "min_child_weight": (1, 5),
    }


def test_unknown_hyperparameter_rejected(env, train_xy) -> None:
    config, _ = env
    _, y = train_xy
    pipe = build_model_pipeline("logistic_regression", {}, config.schema, config.preprocessing, 42, y)
    with pytest.raises(TuningError, match="Unknown hyperparameters"):
        build_search(pipe, {"not_a_param": (1, 2)}, config.tuning, 42)


# ------------------------------------------------------------------------------- search run


def test_run_search_results(env, train_xy) -> None:
    config, _ = env
    x, y = train_xy
    space = config.tuning.search_spaces["xgboost"]
    result = run_search("xgboost", config.models.params["xgboost"], space, config.tuning,
                        config.schema, config.preprocessing, 42, x, y)

    assert result.strategy == "randomized" and result.n_candidates == 3
    assert len(result.trials) == 3
    assert all(result.best_params[k] in space[k] for k in space)
    assert result.trials.iloc[0]["rank"] == 1
    assert result.cv_best_score == pytest.approx(result.trials.iloc[0]["mean_cv_average_precision"])
    assert not any("time" in c for c in result.trials.columns)
    # Refitted on the full train split.
    assert result.best_pipeline.named_steps[PREPROCESS_STEP].n_features_in_ == 29
    assert len(result.best_pipeline.predict_proba(x)) == len(x)


def test_run_search_is_deterministic(env, train_xy) -> None:
    config, _ = env
    x, y = train_xy
    args = ("random_forest", config.models.params["random_forest"], config.tuning.search_spaces["random_forest"],
            config.tuning, config.schema, config.preprocessing, 42, x, y)
    a, b = run_search(*args), run_search(*args)
    pd.testing.assert_frame_equal(a.trials, b.trials)
    assert a.best_params == b.best_params


def test_cv_splitter_only_sees_train_rows(env, train_xy, monkeypatch) -> None:
    config, _ = env
    x, y = train_xy
    seen: list[int] = []

    class RecordingKFold(StratifiedKFold):
        def split(self, X, y=None, groups=None):
            seen.append(len(X))
            return super().split(X, y, groups)

    monkeypatch.setattr(tuning_module, "make_cv", lambda folds, seed: RecordingKFold(folds, shuffle=True, random_state=seed))
    run_search("logistic_regression", config.models.params["logistic_regression"],
               config.tuning.search_spaces["logistic_regression"], config.tuning,
               config.schema, config.preprocessing, 42, x, y)

    assert seen and set(seen) == {len(x)}


def test_cv_results_frame_orders_by_rank() -> None:
    cv = {
        "params": [{"model__C": 0.1}, {"model__C": 1.0}],
        "rank_test_score": np.array([2, 1]),
        "mean_test_score": np.array([0.5, 0.7]),
        "std_test_score": np.array([0.01, 0.02]),
        "split0_test_score": np.array([0.5, 0.7]),
        "split1_test_score": np.array([0.5, 0.7]),
        "mean_fit_time": np.array([1.0, 2.0]),
    }
    frame = cv_results_frame("logistic_regression", cv, ["C"], 2)

    assert frame["trial"].tolist() == [1, 0]
    assert frame["param_C"].tolist() == [1.0, 0.1]
    assert json.loads(frame.loc[0, "params"]) == {"C": 1.0}
    assert "mean_fit_time" not in frame.columns


# ---------------------------------------------------------------------------- selection


def test_rank_candidates_breaks_ties_by_name() -> None:
    assert rank_candidates({"xgboost": 0.8, "random_forest": 0.8, "logistic_regression": 0.9}) == [
        "logistic_regression", "random_forest", "xgboost",
    ]


def test_build_best_model_uses_validation_pr_auc_only() -> None:
    def row(val_pr_auc: float, cv: float) -> dict:
        return {"candidate_file": "x", "best_params": {}, "cv_best_average_precision": cv,
                "val": {"pr_auc": val_pr_auc, "roc_auc": 0.9}}

    # The model with the best CV score is not the validation leader: validation decides.
    best = build_best_model({"a": row(0.70, 0.99), "b": row(0.80, 0.50), "c": row(0.75, 0.60)})

    assert best["leader"] == "b" and best["runner_up"] == "c"
    assert best["ranking_by_val_pr_auc"] == ["b", "c", "a"]
    assert best["margin_over_runner_up"] == pytest.approx(0.05)
    assert "test" in best["not_used"] and "temporal_holdout" in best["not_used"]


def test_load_baseline_val_pr_auc(env, tmp_path) -> None:
    config, _ = env
    assert set(load_baseline_val_pr_auc(config)) == set(CANDIDATE_MODELS)
    missing = dataclasses.replace(config, paths=dataclasses.replace(config.paths, baseline_dir=tmp_path))
    assert load_baseline_val_pr_auc(missing) == {}


# ------------------------------------------------------------------------------- outputs


def test_outputs_generated(env) -> None:
    config, _ = env
    assert all(p.is_file() for p in stage_outputs(config))
    assert {p.name for p in config.paths.candidates_dir.iterdir()} == {"lr.joblib", "rf.joblib", "xgb.joblib"}


def test_summary_content(env) -> None:
    config, _ = env
    summary = _read(config, SUMMARY_FILENAME)

    assert summary["partitions_not_read"] == ["test", "temporal_holdout"]
    assert set(summary["models"]) == set(CANDIDATE_MODELS)
    for name, row in summary["models"].items():
        space = config.tuning.search_spaces[name]
        assert set(row["best_params"]) == set(space)
        assert row["baseline_val_pr_auc"] is not None
        assert row["val_pr_auc_change_vs_baseline"] == pytest.approx(row["val"]["pr_auc"] - row["baseline_val_pr_auc"])
        assert row["improved_over_baseline"] == (row["val_pr_auc_change_vs_baseline"] > 0)
        assert {"pr_auc", "roc_auc", "precision", "recall", "f1", "f2"} <= set(row["val"]) <= set(row["train"]) | set(row["val"])
    assert summary["models"]["logistic_regression"]["search_strategy"] == "grid"


def test_best_model_identifies_leader_and_per_model_best(env) -> None:
    config, _ = env
    best, summary = _read(config, BEST_MODEL_FILENAME), _read(config, SUMMARY_FILENAME)
    val = {m: row["val"]["pr_auc"] for m, row in summary["models"].items()}

    assert best["leader"] == max(val, key=val.get)
    assert set(best["best_per_model"]) == set(CANDIDATE_MODELS)
    for name, row in best["best_per_model"].items():
        assert row["best_params"] == summary["models"][name]["best_params"]


def test_tuning_results_cover_every_trial(env) -> None:
    config, _ = env
    results, summary = _read(config, RESULTS_FILENAME), _read(config, SUMMARY_FILENAME)
    for name, row in summary["models"].items():
        trials = results[results["model"] == name]
        assert len(trials) == row["n_candidates"]
        best = trials.sort_values("rank").iloc[0]
        assert json.loads(best["params"]) == row["best_params"]


def test_saved_candidates_reproduce_reported_metrics(env) -> None:
    config, _ = env
    summary = _read(config, SUMMARY_FILENAME)
    val = pd.read_csv(config.paths.processed_dir / "val.csv")
    x_val, y_val = split_features_target(val, config.schema)
    for name, row in summary["models"].items():
        pipeline = joblib.load(row["candidate_file"])
        proba = pipeline.predict_proba(x_val)[:, 1]
        assert compute_metrics(y_val, proba, 0.5)["pr_auc"] == pytest.approx(row["val"]["pr_auc"])


def test_tuning_is_reproducible(env) -> None:
    config, config_path = env
    before = {p: p.read_bytes() for p in stage_outputs(config)}
    assert main(["--config", str(config_path), "--log-level", "WARNING"]) == 0
    assert {p: p.read_bytes() for p in stage_outputs(config)} == before


# -------------------------------------------------------------------------------- MLflow


def test_mlflow_candidate_and_trial_runs(env) -> None:
    config, _ = env
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    summary = _read(config, SUMMARY_FILENAME)

    candidates = {}
    for run in sorted(runs, key=lambda r: r.info.start_time):
        if run.data.tags.get("run_type") == "candidate":
            candidates[run.data.tags["model"]] = run
    assert set(candidates) == set(CANDIDATE_MODELS)

    for name, run in candidates.items():
        tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
        assert run.info.run_name == name and tags["stage"] == "train"
        for key in ("git_commit", "git_dirty", "data_sha256", "data_dvc_md5", "config_hash"):
            assert key in tags
        assert params["cv_folds"] == "3" and params["n_iter"] == "3" and params["seed"] == "42"
        for param in config.tuning.search_spaces[name]:
            assert f"best_{param}" in params
        assert {"cv_best_average_precision", "val_pr_auc", "val_roc_auc", "val_ref050_f2",
                "train_pr_auc", "val_pr_auc_change_vs_baseline"} <= set(metrics)
        assert not any(k.startswith("test_") or k.startswith("temporal_") for k in metrics)
        artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
        assert {"metrics.json", "tuning_summary.json", "cv_results.json"} <= artifacts
        assert any(a.endswith(".joblib") for a in artifacts)

        children = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == run.info.run_id]
        assert len(children) == summary["models"][name]["n_candidates"]
        assert all(c.data.tags["run_type"] == "tuning_trial" for c in children)
        assert all("cv_mean_average_precision" in c.data.metrics for c in children)


# ------------------------------------------------------------------------------- leakage


def test_never_reads_test_or_holdout(tmp_path, config_dict_factory, transactions_factory, monkeypatch) -> None:
    config_path = _setup(tmp_path, config_dict_factory, transactions_factory, run_baseline=False)
    config = load_config(config_path)
    for name in ("test.csv", "temporal_holdout.csv"):
        (config.paths.processed_dir / name).unlink()
    read_paths: list[str] = []
    real_read_csv = pd.read_csv

    def spy(path, *args, **kwargs):
        read_paths.append(Path(path).name)
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    assert main(["--config", str(config_path), "--log-level", "WARNING"]) == 0
    assert sorted(read_paths) == ["train.csv", "val.csv"]


def test_validation_labels_do_not_influence_hyperparameters(
    tmp_path, config_dict_factory, transactions_factory
) -> None:
    config_path = _setup(tmp_path, config_dict_factory, transactions_factory, run_baseline=False)
    config = load_config(config_path)
    assert main(["--config", str(config_path), "--log-level", "WARNING"]) == 0
    trials_before = _read(config, RESULTS_FILENAME)
    params_before = {m: r["best_params"] for m, r in _read(config, SUMMARY_FILENAME)["models"].items()}

    val_path = config.paths.processed_dir / "val.csv"
    val = pd.read_csv(val_path)
    val["Class"] = val["Class"].sample(frac=1, random_state=0).to_numpy()  # scramble val labels
    val.to_csv(val_path, index=False)
    assert main(["--config", str(config_path), "--log-level", "WARNING"]) == 0

    pd.testing.assert_frame_equal(_read(config, RESULTS_FILENAME), trials_before)
    assert {m: r["best_params"] for m, r in _read(config, SUMMARY_FILENAME)["models"].items()} == params_before


# ----------------------------------------------------------------------------------- CLI


def test_missing_partitions_fail_and_remove_stale_outputs(config_file) -> None:
    config = load_config(config_file)
    stale = stage_outputs(config)[-1]
    stale.parent.mkdir(parents=True)
    stale.write_text("stale")

    assert main(["--config", str(config_file)]) == 1
    assert not stale.exists()


def test_invalid_config_exit_code(tmp_path) -> None:
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_partition_access_is_limited_to_train_and_val(env) -> None:
    config, _ = env
    assert tune_stage.TRAINING_PARTITIONS == ("train", "val")
    for name in ("test", "temporal_holdout"):
        with pytest.raises(tune_stage.TrainingError, match="must not read"):
            tune_stage.partition_path(config, name)
