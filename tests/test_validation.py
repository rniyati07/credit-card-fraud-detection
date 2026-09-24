"""Unit tests for dataset validation and deduplication (DOC-05 M1, §15)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import SchemaConfig, ValidationConfig, load_config
from fraud_detection.data.validate import (
    DataLoadError,
    Status,
    ValidationResult,
    compute_file_sha256,
    load_raw_data,
    validate_dataframe,
    validate_dataset,
    write_report,
)
from fraud_detection.pipeline.validate import main

REPORT_KEYS = {
    "total_rows",
    "total_columns",
    "fraud_count",
    "fraud_rate",
    "missing_values",
    "duplicate_rows",
    "schema_validation_status",
    "overall_status",
    "timestamp",
}


def _validate(
    df: pd.DataFrame, schema: SchemaConfig, validation: ValidationConfig
) -> ValidationResult:
    return validate_dataframe(df, schema, validation)


def _check(result: ValidationResult, name: str) -> dict:
    return next(c for c in result.report["checks"] if c["name"] == name)


# ---------------------------------------------------------------------------- loading


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="not found"):
        load_raw_data(tmp_path / "absent.csv")


def test_load_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DataLoadError, match="empty"):
        load_raw_data(path)


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_bytes(b"a,b\n1,2\n")
    assert compute_file_sha256(path) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()


# -------------------------------------------------------------------------- happy path


def test_valid_dataset_passes(valid_df, schema_config, validation_config) -> None:
    result = _validate(valid_df, schema_config, validation_config)

    report = result.report
    assert result.passed
    assert REPORT_KEYS <= report.keys()
    assert report["overall_status"] == "PASS"
    assert report["schema_validation_status"] == "PASS"
    assert report["total_rows"] == 200
    assert report["total_columns"] == 31
    assert report["fraud_count"] == 10
    assert report["fraud_rate"] == pytest.approx(0.05)
    assert report["missing_values_total"] == 0
    assert report["duplicate_rows"] == 0
    assert report["errors"] == [] and report["warnings"] == []
    assert list(result.clean_data.columns) == list(schema_config.required_columns)


def test_report_is_json_serialisable(valid_df, schema_config, validation_config, tmp_path) -> None:
    result = _validate(valid_df, schema_config, validation_config)
    path = write_report(result.report, tmp_path / "nested" / "report.json")
    assert json.loads(path.read_text())["overall_status"] == "PASS"


def test_validate_dataset_from_csv(valid_df, schema_config, validation_config, tmp_path) -> None:
    path = tmp_path / "creditcard.csv"
    valid_df.to_csv(path, index=False)

    result = validate_dataset(path, schema_config, validation_config)

    assert result.passed
    assert result.report["dataset"]["sha256"] == compute_file_sha256(path)


# ----------------------------------------------------------------------------- schema


def test_missing_required_column_fails(valid_df, schema_config, validation_config) -> None:
    result = _validate(valid_df.drop(columns=["V5"]), schema_config, validation_config)

    assert not result.passed
    assert result.clean_data is None
    assert result.report["schema_validation_status"] == "FAIL"
    assert _check(result, "schema")["details"]["missing_columns"] == ["V5"]
    assert _check(result, "dtypes")["status"] == "SKIPPED"


def test_unexpected_column_warns_and_is_dropped(valid_df, schema_config, validation_config) -> None:
    result = _validate(valid_df.assign(extra="x"), schema_config, validation_config)

    assert result.passed
    assert result.report["schema_validation_status"] == "WARN"
    assert _check(result, "schema")["details"]["unexpected_columns"] == ["extra"]
    assert "extra" not in result.clean_data.columns


def test_non_numeric_feature_fails(valid_df, schema_config, validation_config) -> None:
    df = valid_df.astype({"V3": str})

    result = _validate(df, schema_config, validation_config)

    assert not result.passed
    assert result.report["schema_validation_status"] == "FAIL"
    assert _check(result, "dtypes")["details"]["non_numeric_columns"] == ["V3"]


def test_boolean_feature_fails(valid_df, schema_config, validation_config) -> None:
    df = valid_df.assign(V3=valid_df["V3"] > 0)
    assert not _validate(df, schema_config, validation_config).passed


def test_non_integer_target_fails(valid_df, schema_config, validation_config) -> None:
    df = valid_df.astype({"Class": float})
    df.loc[0, "Class"] = 0.5

    result = _validate(df, schema_config, validation_config)

    assert not result.passed
    assert _check(result, "dtypes")["details"]["target_integer_castable"] is False


# ---------------------------------------------------------------------- missing / inf


def test_missing_target_fails(valid_df, schema_config, validation_config) -> None:
    df = valid_df.astype({"Class": float})
    df.loc[3, "Class"] = np.nan

    result = _validate(df, schema_config, validation_config)

    assert not result.passed
    assert result.report["missing_values"]["Class"] == 1


def test_missing_feature_warns(valid_df, schema_config, validation_config) -> None:
    df = valid_df.copy()
    df.loc[[1, 2], "V7"] = np.nan

    result = _validate(df, schema_config, validation_config)

    assert result.passed
    assert _check(result, "missing_values")["status"] == "WARN"
    assert result.report["missing_values"]["V7"] == 2
    assert result.report["missing_values_total"] == 2


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_values_fail(valid_df, schema_config, validation_config, value) -> None:
    df = valid_df.copy()
    df.loc[0, "V1"] = value

    result = _validate(df, schema_config, validation_config)

    assert not result.passed
    assert result.report["infinite_values_total"] == 1


# ----------------------------------------------------------------------------- target


def test_target_outside_binary_fails(valid_df, schema_config, validation_config) -> None:
    df = valid_df.copy()
    df.loc[0, "Class"] = 2

    result = _validate(df, schema_config, validation_config)

    assert not result.passed
    assert _check(result, "target_values")["details"]["invalid_values"] == [2]
    assert _check(result, "class_distribution")["status"] == "SKIPPED"


def test_single_class_fails(valid_df, schema_config, validation_config) -> None:
    result = _validate(valid_df.assign(Class=0), schema_config, validation_config)

    assert not result.passed
    assert _check(result, "target_values")["status"] == "FAIL"


def test_fraud_rate_outside_expected_range_warns(
    make_transactions, schema_config, validation_config
) -> None:
    df = make_transactions(n_rows=1000, n_fraud=1)  # 0.1% < configured min of 1%

    result = _validate(df, schema_config, validation_config)

    assert result.passed
    assert _check(result, "class_distribution")["status"] == "WARN"
    assert result.report["fraud_count"] == 1


# ----------------------------------------------------------------- rows / value checks


def test_row_count_below_minimum_fails(make_transactions, schema_config, validation_config) -> None:
    result = _validate(make_transactions(n_rows=5, n_fraud=1), schema_config, validation_config)

    assert not result.passed
    assert _check(result, "row_count")["status"] == "FAIL"


@pytest.mark.parametrize("column", ["Amount", "Time"])
def test_negative_values_fail(valid_df, schema_config, validation_config, column) -> None:
    df = valid_df.copy()
    df.loc[0, column] = -1.0

    result = _validate(df, schema_config, validation_config)

    assert not result.passed
    assert _check(result, "negative_values")["details"]["per_column"][column] == 1


def test_outliers_reported_not_removed(valid_df, schema_config, validation_config) -> None:
    df = valid_df.copy()
    df.loc[0, "Amount"] = 1e7

    result = _validate(df, schema_config, validation_config)

    assert result.passed
    assert _check(result, "outliers")["details"]["per_column"]["Amount"]["count"] >= 1
    assert len(result.clean_data) == len(df)
    assert result.clean_data["Amount"].max() == 1e7


# ------------------------------------------------------------------------- duplicates


def test_duplicates_removed_keeping_first(valid_df, schema_config, validation_config) -> None:
    fraud_row = valid_df[valid_df["Class"] == 1].iloc[[0]]
    legit_rows = valid_df[valid_df["Class"] == 0].iloc[[0, 1]]
    df = pd.concat([valid_df, fraud_row, legit_rows, legit_rows.iloc[[0]]], ignore_index=True)

    result = _validate(df, schema_config, validation_config)

    report = result.report
    assert result.passed
    assert _check(result, "duplicates")["status"] == "WARN"
    assert report["duplicate_rows"] == 4
    assert report["duplicate_rows_by_class"] == {"0": 3, "1": 1}
    assert report["clean_dataset"]["total_rows"] == len(valid_df)
    assert report["clean_dataset"]["duplicates_removed"] == 4
    assert report["clean_dataset"]["fraud_count"] == 10
    pd.testing.assert_frame_equal(result.clean_data, valid_df)  # first occurrences, original order


# -------------------------------------------------------------------------- CLI stage


def test_stage_writes_outputs_on_success(config_file, valid_df) -> None:
    config = load_config(config_file)
    config.paths.raw.parent.mkdir(parents=True)
    valid_df.to_csv(config.paths.raw, index=False)

    assert main(["--config", str(config_file)]) == 0

    report = json.loads(config.paths.validation_report.read_text())
    assert report["overall_status"] == "PASS"
    clean = pd.read_csv(config.paths.interim)
    assert len(clean) == len(valid_df)


def test_stage_fails_and_removes_stale_clean_data(config_file, valid_df) -> None:
    config = load_config(config_file)
    config.paths.raw.parent.mkdir(parents=True)
    valid_df.drop(columns=["Class"]).to_csv(config.paths.raw, index=False)
    config.paths.interim.parent.mkdir(parents=True)
    config.paths.interim.write_text("stale")

    assert main(["--config", str(config_file)]) == 1

    assert json.loads(config.paths.validation_report.read_text())["overall_status"] == "FAIL"
    assert not config.paths.interim.exists()


def test_stage_missing_file_writes_fail_report(config_file) -> None:
    config = load_config(config_file)

    assert main(["--config", str(config_file)]) == 1

    report = json.loads(config.paths.validation_report.read_text())
    assert report["overall_status"] == "FAIL"
    assert report["checks"][0]["name"] == "file_load"
    assert report["total_rows"] is None


def test_stage_invalid_config_exit_code(tmp_path) -> None:
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_report_is_written_with_lf_line_endings(valid_df, schema_config, validation_config, tmp_path) -> None:
    # Git-tracked DVC output: identical bytes on every platform (DOC-05 §23 DEV-15).
    result = _validate(valid_df, schema_config, validation_config)
    raw = write_report(result.report, tmp_path / "report.json").read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
