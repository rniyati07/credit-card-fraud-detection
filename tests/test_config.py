"""Unit tests for configuration loading and invariants (DOC-05 §6, §7)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conftest import make_config_dict
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


def test_missing_section_raises(tmp_path: Path) -> None:
    raw = make_config_dict(tmp_path)
    del raw["validation"]
    with pytest.raises(ConfigError, match="validation"):
        parse_config(raw)


def test_time_in_features_raises(tmp_path: Path) -> None:
    raw = make_config_dict(tmp_path)
    raw["schema"]["features"][0] = "Time"
    with pytest.raises(ConfigError, match="must not be a model feature"):
        parse_config(raw)


def test_wrong_feature_count_raises(tmp_path: Path) -> None:
    raw = make_config_dict(tmp_path)
    raw["schema"]["features"].pop()
    with pytest.raises(ConfigError, match="exactly 29"):
        parse_config(raw)


def test_required_columns_mismatch_raises(tmp_path: Path) -> None:
    raw = make_config_dict(tmp_path)
    raw["schema"]["required_columns"].remove("Class")
    with pytest.raises(ConfigError, match="required_columns"):
        parse_config(raw)


def test_invalid_fraud_rate_range_raises(tmp_path: Path) -> None:
    raw = make_config_dict(tmp_path)
    raw["validation"]["expected_fraud_rate"] = {"min": 0.5, "max": 0.1}
    with pytest.raises(ConfigError, match="expected_fraud_rate"):
        parse_config(raw)


def test_config_round_trips_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(make_config_dict(tmp_path)))
    assert load_config(path).schema.target == "Class"
