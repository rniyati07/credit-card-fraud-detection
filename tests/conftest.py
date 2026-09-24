"""Shared synthetic fixtures. Tests never require the real dataset (DOC-03 §9)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_detection.config import (
    PreprocessingConfig,
    SchemaConfig,
    SplitConfig,
    TemporalConfig,
    ValidationConfig,
)

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
REQUIRED_COLUMNS = ["Time", *FEATURES, "Class"]


def make_config_dict(tmp_path: Path) -> dict[str, Any]:
    """Config mapping mirroring config/config.yaml, with paths under ``tmp_path``."""
    return {
        "project": {"name": "test", "seed": 42},
        "paths": {
            "raw": str(tmp_path / "raw" / "creditcard.csv"),
            "interim": str(tmp_path / "interim" / "clean.csv"),
            "validation_report": str(tmp_path / "reports" / "validation" / "validation_report.json"),
            "eda_dir": str(tmp_path / "reports" / "eda"),
            "processed_dir": str(tmp_path / "processed"),
            "split_summary": str(tmp_path / "reports" / "split_summary.json"),
            "temporal_split_summary": str(tmp_path / "reports" / "temporal" / "temporal_split_summary.json"),
        },
        "schema": {
            "target": "Class",
            "positive_label": 1,
            "ordering_column": "Time",
            "features": list(FEATURES),
            "required_columns": list(REQUIRED_COLUMNS),
        },
        "validation": {
            "min_rows": 10,
            "expected_fraud_rate": {"min": 0.01, "max": 0.5},
            "non_negative_columns": ["Time", "Amount"],
            "outlier_iqr_multiplier": 3.0,
        },
        "eda": {
            "top_k_features": 6,
            "time_bin_seconds": 3600,
            "histogram_bins": 30,
            "correlation_top_pairs": 5,
            "figure_dpi": 60,
        },
        "split": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
            "stratify": True,
            "prevalence_tolerance_pp": 0.02,
            "row_id_column": "row_id",
        },
        "temporal": {"holdout_fraction": 0.15, "min_fraud_warning": 20},
        "preprocessing": {"robust_scaled_features": ["Amount"]},
    }


@pytest.fixture
def schema_config() -> SchemaConfig:
    return SchemaConfig(
        target="Class",
        positive_label=1,
        ordering_column="Time",
        features=tuple(FEATURES),
        required_columns=tuple(REQUIRED_COLUMNS),
    )


@pytest.fixture
def validation_config() -> ValidationConfig:
    return ValidationConfig(
        min_rows=10,
        expected_fraud_rate_min=0.01,
        expected_fraud_rate_max=0.5,
        non_negative_columns=("Time", "Amount"),
        outlier_iqr_multiplier=3.0,
    )


@pytest.fixture
def split_config() -> SplitConfig:
    return SplitConfig(
        train=0.70,
        val=0.15,
        test=0.15,
        stratify=True,
        prevalence_tolerance_pp=0.02,
        row_id_column="row_id",
    )


@pytest.fixture
def temporal_config() -> TemporalConfig:
    return TemporalConfig(holdout_fraction=0.15, min_fraud_warning=20)


@pytest.fixture
def preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig(robust_scaled_features=("Amount",))


@pytest.fixture
def make_transactions() -> Callable[..., pd.DataFrame]:
    """Factory for a synthetic dataset with the expected schema and no duplicates."""

    def _make(n_rows: int = 200, n_fraud: int = 10, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        data: dict[str, Any] = {"Time": np.sort(rng.uniform(0, 172_800, n_rows))}
        for i in range(1, 29):
            data[f"V{i}"] = rng.normal(size=n_rows)
        data["Amount"] = rng.lognormal(mean=3.0, sigma=1.0, size=n_rows).round(2)
        labels = np.zeros(n_rows, dtype=np.int64)
        labels[rng.choice(n_rows, size=n_fraud, replace=False)] = 1
        data["Class"] = labels
        return pd.DataFrame(data)

    return _make


@pytest.fixture
def valid_df(make_transactions: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    return make_transactions()


@pytest.fixture
def config_dict(tmp_path: Path) -> dict[str, Any]:
    return make_config_dict(tmp_path)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(make_config_dict(tmp_path)), encoding="utf-8")
    return path
