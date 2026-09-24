"""Descriptive statistics for the EDA (DOC-02 §4).

All functions are pure: they take the deduplicated dataset (or a column of it) and
return plain dicts / DataFrames. Nothing here is fitted, and no result feeds the
modelling pipeline. EDA informs documented decisions only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.config import EdaConfig, SchemaConfig

logger = logging.getLogger(__name__)

AMOUNT_COLUMN = "Amount"


class EdaInputError(Exception):
    """Raised when the EDA input dataset is missing or unusable."""


@dataclass(frozen=True)
class EdaResults:
    """All computed EDA results, used for figures, the Markdown report and the JSON summary."""

    overview: dict[str, Any]
    data_quality: dict[str, Any]
    class_distribution: dict[str, Any]
    amount_statistics: pd.DataFrame
    time_summary: dict[str, Any]
    fraud_rate_over_time: pd.DataFrame
    feature_summary: pd.DataFrame
    variance: dict[str, Any]
    mean_differences: pd.DataFrame
    target_correlation: pd.Series
    feature_correlation: pd.DataFrame
    top_correlated_pairs: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the results (large matrices summarised)."""
        return {
            "overview": self.overview,
            "data_quality": self.data_quality,
            "class_distribution": self.class_distribution,
            "amount_statistics": self.amount_statistics.to_dict(orient="index"),
            "time_summary": self.time_summary,
            "feature_summary": self.feature_summary.to_dict(orient="index"),
            "variance": self.variance,
            "mean_differences": self.mean_differences.to_dict(orient="index"),
            "target_correlation": self.target_correlation.to_dict(),
            "top_correlated_pairs": self.top_correlated_pairs.to_dict(orient="records"),
        }


def load_clean_data(path: str | Path) -> pd.DataFrame:
    """Load the validated, deduplicated dataset produced by the ``validate`` stage.

    Raises:
        EdaInputError: If the file is missing or unreadable.
    """
    path = Path(path)
    if not path.is_file():
        raise EdaInputError(
            f"Clean dataset not found at '{path}'. "
            "Run `python -m fraud_detection.pipeline.validate` first."
        )
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        raise EdaInputError(f"Clean dataset at '{path}' could not be read: {exc}") from exc
    logger.info("Loaded %s: %d rows x %d columns", path, df.shape[0], df.shape[1])
    return df


def v_features(schema: SchemaConfig) -> list[str]:
    """The anonymised PCA components (all configured features except ``Amount``)."""
    return [f for f in schema.features if f != AMOUNT_COLUMN]


# ------------------------------------------------------------------ overview / quality


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Shape, data types and in-memory size of the dataset."""
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    dtype_counts: dict[str, int] = {}
    for dtype in dtypes.values():
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    memory_bytes = int(df.memory_usage(deep=True).sum())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(map(str, df.columns)),
        "dtypes": dtypes,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "memory_bytes": memory_bytes,
        "memory_mb": memory_bytes / 1024**2,
    }


def data_quality(df: pd.DataFrame, validation_report: dict[str, Any] | None) -> dict[str, Any]:
    """Re-check duplicates / missing / infinite values and reconcile with the M1 report."""
    numeric = df.select_dtypes(include="number")
    quality: dict[str, Any] = {
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_total": int(df.isna().sum().sum()),
        "infinite_values_total": int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
        "validation_report_available": validation_report is not None,
    }
    if validation_report is not None:
        clean = validation_report.get("clean_dataset") or {}
        quality.update(
            {
                "m1_overall_status": validation_report.get("overall_status"),
                "m1_raw_rows": validation_report.get("total_rows"),
                "m1_raw_fraud_count": validation_report.get("fraud_count"),
                "m1_duplicates_removed": validation_report.get("duplicate_rows"),
                "m1_clean_rows": clean.get("total_rows"),
                "m1_clean_fraud_count": clean.get("fraud_count"),
                "m1_raw_sha256": (validation_report.get("dataset") or {}).get("sha256"),
            }
        )
        quality["matches_m1_report"] = bool(
            quality["m1_overall_status"] == "PASS" and quality["m1_clean_rows"] == len(df)
        )
    return quality


# ------------------------------------------------------------------------ target / amount


def class_distribution(target: pd.Series, positive_label: int) -> dict[str, Any]:
    """Class counts, fraud percentage and imbalance ratio (legitimate : fraud)."""
    total = int(len(target))
    fraud = int((target == positive_label).sum())
    legit = total - fraud
    return {
        "total": total,
        "fraud_count": fraud,
        "legitimate_count": legit,
        "fraud_rate": fraud / total if total else 0.0,
        "fraud_percentage": 100 * fraud / total if total else 0.0,
        "imbalance_ratio": legit / fraud if fraud else float("inf"),
        # Accuracy of a constant "legitimate" predictor: why accuracy misleads (DOC-02 §8).
        "all_legitimate_accuracy": legit / total if total else 0.0,
    }


def describe_amount(amount: pd.Series) -> dict[str, float]:
    """Distribution statistics for a transaction-amount series."""
    if amount.empty:
        return {}
    q = amount.quantile([0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(amount.size),
        "mean": float(amount.mean()),
        "std": float(amount.std()),
        "min": float(amount.min()),
        "q25": float(q[0.25]),
        "median": float(q[0.5]),
        "q75": float(q[0.75]),
        "q95": float(q[0.95]),
        "q99": float(q[0.99]),
        "max": float(amount.max()),
        "skewness": float(amount.skew()),
        "log1p_skewness": float(np.log1p(amount).skew()),
        "zero_share": float((amount == 0).mean()),
        "le_one_share": float((amount <= 1).mean()),
    }


def amount_statistics(df: pd.DataFrame, target: str, positive_label: int) -> pd.DataFrame:
    """``describe_amount`` for all transactions, legitimate only and fraud only."""
    is_fraud = df[target] == positive_label
    groups = {
        "all": df[AMOUNT_COLUMN],
        "legitimate": df.loc[~is_fraud, AMOUNT_COLUMN],
        "fraud": df.loc[is_fraud, AMOUNT_COLUMN],
    }
    return pd.DataFrame({name: describe_amount(s) for name, s in groups.items()}).T


# ---------------------------------------------------------------------------------- time


def fraud_rate_over_time(
    time: pd.Series, target: pd.Series, bin_seconds: int, positive_label: int
) -> pd.DataFrame:
    """Transactions, frauds and fraud rate per fixed-width bin of elapsed ``Time``.

    Descriptive only: bins are elapsed-time windows, not hour-of-day features.
    """
    bins = (time // bin_seconds).astype(int)
    grouped = (target == positive_label).groupby(bins)
    out = pd.DataFrame({"transactions": grouped.size(), "frauds": grouped.sum().astype(int)})
    out["fraud_rate"] = out["frauds"] / out["transactions"]
    out.index = out.index * bin_seconds / 3600.0
    out.index.name = "bin_start_hours"
    return out


def time_summary(
    df: pd.DataFrame, ordering_column: str, target: str, positive_label: int, rate: pd.DataFrame
) -> dict[str, Any]:
    """Time span, monotonicity and fraud-rate variation over elapsed time."""
    time = df[ordering_column]
    is_fraud = df[target] == positive_label
    return {
        "min_seconds": float(time.min()),
        "max_seconds": float(time.max()),
        "span_hours": float((time.max() - time.min()) / 3600),
        "is_sorted_in_file": bool(time.is_monotonic_increasing),
        "legitimate_median_hours": float(time[~is_fraud].median() / 3600),
        "fraud_median_hours": float(time[is_fraud].median() / 3600),
        "n_bins": int(len(rate)),
        "bin_fraud_rate_min": float(rate["fraud_rate"].min()),
        "bin_fraud_rate_max": float(rate["fraud_rate"].max()),
        "bin_fraud_rate_median": float(rate["fraud_rate"].median()),
        "bins_without_fraud": int((rate["frauds"] == 0).sum()),
    }


# ------------------------------------------------------------------------------ features


def feature_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Per-feature mean, std, variance, range, quartiles, skewness and kurtosis."""
    data = df[features]
    summary = data.describe().T.rename(columns={"25%": "q25", "50%": "median", "75%": "q75"})
    summary["variance"] = data.var()
    summary["skewness"] = data.skew()
    summary["kurtosis"] = data.kurt()
    return summary.drop(columns="count")


def variance_analysis(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """Variance profile of the PCA components and their scale relative to ``Amount``."""
    variances = df[features].var()
    result: dict[str, Any] = {
        "variances": {k: float(v) for k, v in variances.items()},
        "max_feature": str(variances.idxmax()),
        "min_feature": str(variances.idxmin()),
        "max_to_min_ratio": float(variances.max() / variances.min()),
        # PCA outputs are ordered by explained variance; check whether dedup preserved that.
        "monotonic_non_increasing": bool(variances.is_monotonic_decreasing),
    }
    if AMOUNT_COLUMN in df.columns:
        amount_var = float(df[AMOUNT_COLUMN].var())
        result["amount_variance"] = amount_var
        result["amount_to_max_v_variance_ratio"] = amount_var / float(variances.max())
    return result


def class_mean_differences(
    df: pd.DataFrame, features: list[str], target: str, positive_label: int
) -> pd.DataFrame:
    """Class-wise means and standardized mean difference, ranked by absolute effect.

    ``std_mean_diff = (mean_fraud - mean_legitimate) / std_all``.
    """
    is_fraud = df[target] == positive_label
    mean_fraud = df.loc[is_fraud, features].mean()
    mean_legit = df.loc[~is_fraud, features].mean()
    out = pd.DataFrame(
        {
            "mean_legitimate": mean_legit,
            "mean_fraud": mean_fraud,
            "std_mean_diff": (mean_fraud - mean_legit) / df[features].std(),
        }
    )
    order = out["std_mean_diff"].abs().sort_values(ascending=False, kind="mergesort").index
    return out.loc[order]


def target_correlation(df: pd.DataFrame, features: list[str], target: str) -> pd.Series:
    """Pearson (point-biserial) correlation of each feature with the target, by |r|."""
    corr = df[features].corrwith(df[target].astype(float))
    return corr.reindex(corr.abs().sort_values(ascending=False, kind="mergesort").index)


def top_correlated_pairs(corr: pd.DataFrame, k: int) -> pd.DataFrame:
    """The ``k`` feature pairs with the largest absolute correlation (upper triangle)."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("correlation").reset_index()
    pairs.columns = ["feature_a", "feature_b", "correlation"]
    order = pairs["correlation"].abs().sort_values(ascending=False, kind="mergesort").index
    return pairs.loc[order].head(k).reset_index(drop=True)


# -------------------------------------------------------------------------- orchestration


def compute_eda(
    df: pd.DataFrame,
    schema: SchemaConfig,
    eda: EdaConfig,
    validation_report: dict[str, Any] | None = None,
) -> EdaResults:
    """Compute every DOC-02 §4 analysis on the deduplicated dataset."""
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise EdaInputError(f"Clean dataset is missing required columns: {missing}")

    target, pos = schema.target, schema.positive_label
    v_cols = v_features(schema)
    features = list(schema.features)

    rate = fraud_rate_over_time(df[schema.ordering_column], df[target], eda.time_bin_seconds, pos)
    corr = df[features].corr()
    results = EdaResults(
        overview=dataset_overview(df),
        data_quality=data_quality(df, validation_report),
        class_distribution=class_distribution(df[target], pos),
        amount_statistics=amount_statistics(df, target, pos),
        time_summary=time_summary(df, schema.ordering_column, target, pos, rate),
        fraud_rate_over_time=rate,
        feature_summary=feature_summary(df, v_cols),
        variance=variance_analysis(df, v_cols),
        mean_differences=class_mean_differences(df, v_cols, target, pos),
        target_correlation=target_correlation(df, features, target),
        feature_correlation=corr,
        top_correlated_pairs=top_correlated_pairs(corr, eda.correlation_top_pairs),
    )
    logger.info(
        "EDA computed: %d rows, %d frauds (%.4f%%)",
        results.class_distribution["total"],
        results.class_distribution["fraud_count"],
        results.class_distribution["fraud_percentage"],
    )
    return results
