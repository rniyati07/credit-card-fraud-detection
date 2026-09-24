"""Unit tests for the EDA utilities and end-to-end run (DOC-05 M2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import load_config
from fraud_detection.eda.analysis import (
    EdaInputError,
    class_distribution,
    class_mean_differences,
    compute_eda,
    data_quality,
    dataset_overview,
    describe_amount,
    fraud_rate_over_time,
    load_clean_data,
    target_correlation,
    top_correlated_pairs,
    variance_analysis,
)
from fraud_detection.eda.cli import main, run_eda
from fraud_detection.eda.plots import MAX_FIGURES
from fraud_detection.eda.report import markdown_table

REPORT_SECTIONS = [
    "## Executive summary",
    "## Key findings",
    "## 1. Dataset overview",
    "## 2. Data quality",
    "## 3. Class distribution and fraud imbalance",
    "## 4. Transaction amount behaviour",
    "## 6. Feature observations",
    "## 7. Risks and limitations",
    "## 8. Recommendations for M3",
]


# ------------------------------------------------------------------------------ analysis


def test_class_distribution_counts_and_ratios() -> None:
    result = class_distribution(pd.Series([0, 0, 0, 1]), positive_label=1)

    assert result["fraud_count"] == 1
    assert result["legitimate_count"] == 3
    assert result["fraud_percentage"] == pytest.approx(25.0)
    assert result["imbalance_ratio"] == pytest.approx(3.0)
    assert result["all_legitimate_accuracy"] == pytest.approx(0.75)


def test_describe_amount_known_values() -> None:
    stats = describe_amount(pd.Series([0.0, 1.0, 2.0, 3.0, 4.0]))

    assert stats["count"] == 5
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["min"] == 0.0 and stats["max"] == 4.0
    assert stats["zero_share"] == pytest.approx(0.2)


def test_fraud_rate_over_time_bins() -> None:
    time = pd.Series([0, 100, 3599, 3600, 7300])
    target = pd.Series([1, 0, 0, 1, 1])

    rate = fraud_rate_over_time(time, target, bin_seconds=3600, positive_label=1)

    assert list(rate.index) == [0.0, 1.0, 2.0]
    assert rate["transactions"].tolist() == [3, 1, 1]
    assert rate["frauds"].tolist() == [1, 1, 1]
    assert rate["fraud_rate"].tolist() == pytest.approx([1 / 3, 1.0, 1.0])


def test_class_mean_differences_ranked_by_absolute_effect() -> None:
    df = pd.DataFrame(
        {
            "A": [0.0, 2.0, 0.5, 1.5],  # equal class means: no shift
            "B": [0.0, 0.0, -10.0, -10.0],  # large negative shift
            "Class": [0, 0, 1, 1],
        }
    )

    result = class_mean_differences(df, ["A", "B"], "Class", 1)

    assert list(result.index) == ["B", "A"]
    assert result.loc["B", "mean_fraud"] == -10.0
    assert result.loc["B", "std_mean_diff"] == pytest.approx(-10.0 / df["B"].std())
    assert result.loc["A", "std_mean_diff"] == pytest.approx(0.0)


def test_target_correlation_sorted_by_absolute_value() -> None:
    df = pd.DataFrame(
        {"weak": [0.1, 0.0, 0.2, 0.1], "strong_neg": [1.0, 1.0, 0.0, 0.0], "Class": [0, 0, 1, 1]}
    )

    corr = target_correlation(df, ["weak", "strong_neg"], "Class")

    assert corr.index[0] == "strong_neg"
    assert corr["strong_neg"] == pytest.approx(-1.0)


def test_top_correlated_pairs_upper_triangle_only() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.9, -0.5], [0.9, 1.0, 0.1], [-0.5, 0.1, 1.0]],
        index=list("abc"),
        columns=list("abc"),
    )

    pairs = top_correlated_pairs(corr, k=2)

    assert pairs[["feature_a", "feature_b"]].values.tolist() == [["a", "b"], ["a", "c"]]
    assert pairs["correlation"].tolist() == [0.9, -0.5]


def test_variance_analysis_flags_monotonic_order() -> None:
    df = pd.DataFrame({"V1": [0, 4, -4, 0], "V2": [0, 2, -2, 0], "Amount": [0, 100, -100, 0]})

    result = variance_analysis(df, ["V1", "V2"])

    assert result["monotonic_non_increasing"] is True
    assert result["max_feature"] == "V1" and result["min_feature"] == "V2"
    assert result["max_to_min_ratio"] == pytest.approx(4.0)
    assert result["amount_to_max_v_variance_ratio"] == pytest.approx(625.0)


def test_dataset_overview(valid_df) -> None:
    overview = dataset_overview(valid_df)

    assert overview["rows"] == 200 and overview["columns"] == 31
    assert overview["dtype_counts"] == {"float64": 30, "int64": 1}
    assert overview["memory_bytes"] > 0


def test_data_quality_reconciles_with_m1_report(valid_df) -> None:
    report = {"overall_status": "PASS", "clean_dataset": {"total_rows": 200, "fraud_count": 10}}

    assert data_quality(valid_df, report)["matches_m1_report"] is True
    report["clean_dataset"]["total_rows"] = 199
    assert data_quality(valid_df, report)["matches_m1_report"] is False
    assert data_quality(valid_df, None)["validation_report_available"] is False


def test_data_quality_detects_residual_duplicates(valid_df) -> None:
    df = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
    assert data_quality(df, None)["duplicate_rows"] == 1


def test_load_clean_data_missing_file(tmp_path: Path) -> None:
    with pytest.raises(EdaInputError, match="pipeline.validate"):
        load_clean_data(tmp_path / "clean.csv")


def test_compute_eda_rejects_missing_columns(valid_df, schema_config, config_file) -> None:
    eda_config = load_config(config_file).eda
    with pytest.raises(EdaInputError, match="missing required columns"):
        compute_eda(valid_df.drop(columns=["V1"]), schema_config, eda_config)


def test_markdown_table() -> None:
    assert markdown_table(["a", "b"], [[1, 2]]) == "| a | b |\n|---|---|\n| 1 | 2 |"


# ----------------------------------------------------------------------------- end to end


def _write_clean(config_file: Path, df: pd.DataFrame) -> None:
    config = load_config(config_file)
    config.paths.interim.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.paths.interim, index=False)


def test_run_eda_writes_outputs(config_file, valid_df) -> None:
    _write_clean(config_file, valid_df)
    config = load_config(config_file)

    outputs = run_eda(config)

    assert 0 < len(outputs.figures) <= MAX_FIGURES
    assert all(p.is_file() and p.parent == config.paths.eda_dir / "figures" for p in outputs.figures.values())
    report = outputs.report.read_text(encoding="utf-8")
    for section in REPORT_SECTIONS:
        assert section in report
    for placeholder in ("TODO", "TBD", "<placeholder>", "nan", "None"):
        assert placeholder not in report
    for name in outputs.figures:
        assert f"](figures/{name}.png)" in report
    summary = json.loads(outputs.summary.read_text(encoding="utf-8"))
    assert summary["class_distribution"]["fraud_count"] == 10
    assert summary["data_quality"]["validation_report_available"] is False


def test_run_eda_is_reproducible(config_file, valid_df) -> None:
    _write_clean(config_file, valid_df)
    config = load_config(config_file)

    first = run_eda(config)
    report_1, summary_1 = first.report.read_bytes(), first.summary.read_bytes()
    second = run_eda(config)

    assert second.report.read_bytes() == report_1
    assert second.summary.read_bytes() == summary_1


def test_main_exit_codes(config_file, tmp_path) -> None:
    assert main(["--config", str(config_file)]) == 1  # clean data not generated yet
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_main_success(config_file, valid_df) -> None:
    _write_clean(config_file, valid_df.assign(V1=np.round(valid_df["V1"], 6)))
    assert main(["--config", str(config_file), "--log-level", "WARNING"]) == 0
