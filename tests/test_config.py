"""Unit tests for configuration loading and invariants (DOC-05 §6, §7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from fraud_detection.config import ConfigError, load_config, parse_config


def test_repository_config_is_valid() -> None:
    config = load_config(Path(__file__).parents[1] / "config" / "config.yaml")

    assert len(config.schema.features) == 29
    assert "Time" not in config.schema.features
    assert config.paths.validation_report == Path("reports/validation/validation_report.json")


def test_load_config_from_file(config_file: Path) -> None:
    assert load_config(config_file).project.seed == 42


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("project: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_missing_section_raises(config_dict: dict[str, Any]) -> None:
    raw = config_dict
    del raw["validation"]
    with pytest.raises(ConfigError, match="validation"):
        parse_config(raw)


def test_time_in_features_raises(config_dict: dict[str, Any]) -> None:
    raw = config_dict
    raw["schema"]["features"][0] = "Time"
    with pytest.raises(ConfigError, match="must not be a model feature"):
        parse_config(raw)


def test_wrong_feature_count_raises(config_dict: dict[str, Any]) -> None:
    raw = config_dict
    raw["schema"]["features"].pop()
    with pytest.raises(ConfigError, match="exactly 29"):
        parse_config(raw)


def test_required_columns_mismatch_raises(config_dict: dict[str, Any]) -> None:
    raw = config_dict
    raw["schema"]["required_columns"].remove("Class")
    with pytest.raises(ConfigError, match="required_columns"):
        parse_config(raw)


def test_invalid_fraud_rate_range_raises(config_dict: dict[str, Any]) -> None:
    raw = config_dict
    raw["validation"]["expected_fraud_rate"] = {"min": 0.5, "max": 0.1}
    with pytest.raises(ConfigError, match="expected_fraud_rate"):
        parse_config(raw)


def test_invalid_eda_setting_raises(config_dict: dict[str, Any]) -> None:
    raw = config_dict
    raw["eda"]["top_k_features"] = 0
    with pytest.raises(ConfigError, match="eda.top_k_features"):
        parse_config(raw)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("split", "train", 0.8, "sum to 1"),
        ("split", "val", 0.0, r"split.val must be in \(0, 1\)"),
        ("split", "stratify", "yes", "must be a boolean"),
        ("split", "prevalence_tolerance_pp", 0, "prevalence_tolerance_pp"),
        ("split", "row_id_column", "Time", "row_id_column"),
        ("temporal", "holdout_fraction", 1.0, "holdout_fraction"),
        ("temporal", "min_fraud_warning", -1, "min_fraud_warning"),
        ("preprocessing", "robust_scaled_features", ["Time"], "robust_scaled_features"),
    ],
)
def test_invalid_m3_settings_raise(config_dict: dict[str, Any], section, key, value, message) -> None:
    config_dict[section][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(config_dict)


def test_repository_config_m3_values() -> None:
    config = load_config(Path(__file__).parents[1] / "config" / "config.yaml")

    assert (config.split.train, config.split.val, config.split.test) == (0.70, 0.15, 0.15)
    assert config.split.stratify is True
    assert config.temporal.holdout_fraction == 0.15
    assert config.preprocessing.robust_scaled_features == ("Amount",)


def test_config_round_trips_through_yaml(tmp_path: Path, config_dict: dict[str, Any]) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config_dict))
    assert load_config(path).schema.target == "Class"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["imbalance"].update(strategy="smote"), "imbalance.strategy"),
        (lambda c: c["models"].update(enabled=["logistic_regression", "svm"]), "unknown models"),
        (lambda c: c["models"].update(enabled=[]), "at least one model"),
        (lambda c: c["models"]["xgboost"].update(scale_pos_weight=10), "scale_pos_weight is set in code"),
        (lambda c: c["models"]["random_forest"].update(random_state=1), "random_state is set in code"),
        (lambda c: c["models"]["random_forest"].pop("class_weight"), "class_weight is required"),
        (lambda c: c["threshold"].update(reference_threshold=1.0), "reference_threshold"),
        (lambda c: c["mlflow"].update(experiment_name=""), "non-empty"),
    ],
)
def test_invalid_m4_settings_raise(config_dict: dict[str, Any], mutate, message) -> None:
    mutate(config_dict)
    with pytest.raises(ConfigError, match=message):
        parse_config(config_dict)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["tuning"].update(n_iter=11), r"n_iter must be in \[1, 10\]"),
        (lambda c: c["tuning"].update(cv_folds=1), "cv_folds must be >= 2"),
        (lambda c: c["tuning"].update(scoring="roc_auc"), "scoring must be 'average_precision'"),
        (lambda c: c["tuning"]["search_spaces"].pop("xgboost"), "missing: \['xgboost'\]"),
        (lambda c: c["tuning"]["search_spaces"].update(svm={"C": [1]}), "extra: \['svm'\]"),
        (lambda c: c["tuning"]["search_spaces"]["random_forest"].update(class_weight=["balanced"]), "cannot be tuned"),
        (lambda c: c["tuning"]["search_spaces"]["xgboost"].update(scale_pos_weight=[1, 10]), "cannot be tuned"),
        (lambda c: c["tuning"]["search_spaces"]["random_forest"].update(n_jobs=[1, 2]), "already fixed"),
        (lambda c: c["tuning"]["search_spaces"]["logistic_regression"].update(C=[]), "non-empty list"),
        (lambda c: c["tuning"]["search_spaces"]["logistic_regression"].update(C=[1, 1]), "duplicate"),
        (lambda c: c["tuning"]["search_spaces"]["logistic_regression"].update(C=[[1]]), "scalars"),
    ],
)
def test_invalid_m5_settings_raise(config_dict: dict[str, Any], mutate, message) -> None:
    mutate(config_dict)
    with pytest.raises(ConfigError, match=message):
        parse_config(config_dict)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["project"].update(model_version=""), "model_version"),
        (lambda c: c["threshold"].update(r_min=0), r"r_min must be in \(0, 1\]"),
        (lambda c: c["threshold"].update(r_min=1.2), r"r_min must be in \(0, 1\]"),
        (lambda c: c["threshold"]["grid"].update(start=0.0), "0 < start <= stop < 1"),
        (lambda c: c["threshold"]["grid"].update(stop=1.0), "0 < start <= stop < 1"),
        (lambda c: c["threshold"]["grid"].update(step=0), "step must be > 0"),
        (lambda c: c["threshold"].update(include_pr_curve_points="yes"), "must be a boolean"),
        (lambda c: c["threshold"].update(r_min_confirmed="yes"), "r_min_confirmed' must be a boolean"),
        (lambda c: c["threshold"].update(fallback="f1"), "fallback must be one of"),
        (lambda c: c["selection"].update(pr_auc_tie_tolerance=-0.1), "pr_auc_tie_tolerance"),
        (lambda c: c.pop("selection"), "selection"),
    ],
)
def test_invalid_m6_settings_raise(config_dict: dict[str, Any], mutate, message) -> None:
    mutate(config_dict)
    with pytest.raises(ConfigError, match=message):
        parse_config(config_dict)


def test_repository_config_m6_values() -> None:
    config = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
    t = config.threshold
    assert (t.r_min, t.grid_start, t.grid_stop, t.grid_step) == (0.80, 0.01, 0.99, 0.01)
    assert t.include_pr_curve_points is True and t.fallback == "f2" and t.reference_threshold == 0.5
    assert t.r_min_confirmed is True  # confirmed after M6 review (DOC-05 §23 DEC-01)
    assert config.selection.pr_auc_tie_tolerance == 0.01
    assert config.project.model_version == "1.0.0"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["evaluation"]["bootstrap"].update(enabled="yes"), "enabled' must be a boolean"),
        (lambda c: c["evaluation"]["bootstrap"].update(n=10), "n must be >= 100"),
        (lambda c: c["evaluation"]["bootstrap"].update(confidence=1.0), "confidence"),
        (lambda c: c["evaluation"].update(inconclusive_fraud_cases=-1), "inconclusive_fraud_cases"),
        (lambda c: c.pop("evaluation"), "evaluation"),
    ],
)
def test_invalid_m7_settings_raise(config_dict: dict[str, Any], mutate, message) -> None:
    mutate(config_dict)
    with pytest.raises(ConfigError, match=message):
        parse_config(config_dict)


def test_repository_config_m7_values() -> None:
    config = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
    e = config.evaluation
    assert (e.bootstrap_enabled, e.bootstrap_n, e.bootstrap_confidence, e.inconclusive_fraud_cases) == (True, 1000, 0.95, 2)
    assert config.paths.test_metrics == Path("reports/test_metrics.json")
    assert config.paths.temporal_dir == Path("reports/temporal")
