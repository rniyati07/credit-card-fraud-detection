"""Dataset validation and deduplication (DOC-02 §3, DOC-05 M1).

Each check is a small, pure function returning a :class:`CheckResult`. Checks never
raise for data problems; they report ``FAIL`` (hard fail) or ``WARN`` so that one
run produces a complete report. Only an absent or unreadable file raises
(:class:`DataLoadError`, FR-001), because nothing else can be checked without data.

Rules implemented (DOC-02 §3):

============================  ===========================================================
Check                         On failure
============================  ===========================================================
Required columns              hard fail
Unexpected columns            warn; excluded from the clean dataset
Row count                     hard fail if below ``validation.min_rows``
Data types                    hard fail (non-target columns numeric; target integer-like)
Missing values                hard fail in target; warn in other columns
Infinite values               hard fail
Target values                 hard fail unless values are {0, 1} and both are present
Class distribution            warn if fraud rate is outside the expected range
Negative values               hard fail for ``validation.non_negative_columns``
Outliers (IQR)                report only; rows are never removed
Duplicate rows                warn; exact duplicates dropped (first kept) before splitting
============================  ===========================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.config import SchemaConfig, ValidationConfig

logger = logging.getLogger(__name__)

ALLOWED_TARGET_VALUES = frozenset({0, 1})


class Status(StrEnum):
    """Outcome of a single check, or of the whole validation run."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class DataValidationError(Exception):
    """Base class for data-validation errors."""


class DataLoadError(DataValidationError):
    """Raised when the dataset file is absent, empty or unreadable (FR-001)."""


@dataclass(frozen=True)
class CheckResult:
    """Result of one validation check."""

    name: str
    status: Status
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Validation report plus the deduplicated dataset (``None`` when validation failed)."""

    report: dict[str, Any]
    clean_data: pd.DataFrame | None

    @property
    def passed(self) -> bool:
        """True when no check hard-failed."""
        return self.report["overall_status"] == Status.PASS


# --------------------------------------------------------------------------- loading


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw transactions CSV with default pandas parsing.

    Default parsing (no index column, float64 for decimals) is required so that exact
    duplicate detection compares the values as stored in the file.

    Raises:
        DataLoadError: If the file does not exist, is empty or cannot be parsed.
    """
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(
            f"Dataset not found at '{path}'. Run `dvc pull` or place the CSV at this path."
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"Dataset at '{path}' is empty") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DataLoadError(f"Dataset at '{path}' could not be read: {exc}") from exc

    logger.info("Loaded %s: %d rows x %d columns", path, df.shape[0], df.shape[1])
    return df


def compute_file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks (DOC-02 §16)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------- checks


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _class_counts(target: pd.Series) -> dict[str, int]:
    counts = target.value_counts(dropna=True).sort_index()
    return {str(_to_builtin(label)): int(n) for label, n in counts.items()}


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars to Python types; render integral floats as ints."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def check_schema(df: pd.DataFrame, required_columns: Sequence[str]) -> CheckResult:
    """Verify all required columns exist; report (warn on) unexpected columns."""
    missing = [c for c in required_columns if c not in df.columns]
    unexpected = [str(c) for c in df.columns if c not in required_columns]
    details = {
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "column_order_matches": list(df.columns) == list(required_columns),
    }
    if missing:
        return CheckResult("schema", Status.FAIL, f"Missing required columns: {missing}", details)
    if unexpected:
        return CheckResult(
            "schema",
            Status.WARN,
            f"Unexpected columns will be excluded from the clean dataset: {unexpected}",
            details,
        )
    return CheckResult("schema", Status.PASS, "All required columns present", details)


def check_row_count(df: pd.DataFrame, min_rows: int) -> CheckResult:
    """Verify the dataset has at least ``min_rows`` rows."""
    n_rows = len(df)
    details = {"total_rows": n_rows, "min_rows": min_rows}
    if n_rows < min_rows:
        return CheckResult(
            "row_count", Status.FAIL, f"Dataset has {n_rows} rows; expected >= {min_rows}", details
        )
    return CheckResult("row_count", Status.PASS, f"Dataset has {n_rows} rows", details)


def check_dtypes(df: pd.DataFrame, schema: SchemaConfig) -> CheckResult:
    """Verify non-target columns are numeric and the target is integer-castable."""
    numeric_columns = [c for c in schema.required_columns if c != schema.target]
    non_numeric = [c for c in numeric_columns if not _is_numeric(df[c])]

    target = df[schema.target]
    target_ok = _is_numeric(target)
    if target_ok:
        values = target.dropna().to_numpy(dtype=float)
        target_ok = bool(np.isfinite(values).all() and (values == np.floor(values)).all())

    details = {
        "column_dtypes": {c: str(df[c].dtype) for c in schema.required_columns},
        "non_numeric_columns": non_numeric,
        "target_integer_castable": target_ok,
    }
    problems = []
    if non_numeric:
        problems.append(f"non-numeric columns: {non_numeric}")
    if not target_ok:
        problems.append(f"target '{schema.target}' is not integer-castable")
    if problems:
        return CheckResult("dtypes", Status.FAIL, "Invalid data types: " + "; ".join(problems), details)
    return CheckResult("dtypes", Status.PASS, "All columns have valid data types", details)


def check_missing_values(df: pd.DataFrame, columns: Sequence[str], target: str) -> CheckResult:
    """Count missing values per column; fail if the target has any, warn otherwise."""
    per_column = {c: int(df[c].isna().sum()) for c in columns}
    with_missing = {c: n for c, n in per_column.items() if n}
    details = {"per_column": per_column, "total": sum(per_column.values())}

    if per_column.get(target, 0):
        return CheckResult(
            "missing_values",
            Status.FAIL,
            f"Target '{target}' has {per_column[target]} missing values",
            details,
        )
    if with_missing:
        return CheckResult(
            "missing_values",
            Status.WARN,
            f"Missing feature values (handled by the pipeline imputer): {with_missing}",
            details,
        )
    return CheckResult("missing_values", Status.PASS, "No missing values", details)


def check_infinite_values(df: pd.DataFrame, columns: Sequence[str]) -> CheckResult:
    """Fail if any column contains +/-inf (indicates corrupt data)."""
    per_column = {c: int(np.isinf(df[c].to_numpy(dtype=float)).sum()) for c in columns}
    with_inf = {c: n for c, n in per_column.items() if n}
    details = {"per_column": per_column, "total": sum(per_column.values())}
    if with_inf:
        return CheckResult("infinite_values", Status.FAIL, f"Infinite values found: {with_inf}", details)
    return CheckResult("infinite_values", Status.PASS, "No infinite values", details)


def check_target_values(df: pd.DataFrame, target: str) -> CheckResult:
    """Verify the target contains only {0, 1} and that both classes are present."""
    observed = sorted(_to_builtin(v) for v in df[target].dropna().unique())
    invalid = [v for v in observed if v not in ALLOWED_TARGET_VALUES]
    details = {"observed_values": observed, "invalid_values": invalid}

    if invalid:
        return CheckResult(
            "target_values", Status.FAIL, f"Target has values outside {{0, 1}}: {invalid}", details
        )
    if set(observed) != ALLOWED_TARGET_VALUES:
        return CheckResult(
            "target_values", Status.FAIL, f"Both classes must be present; found {observed}", details
        )
    return CheckResult("target_values", Status.PASS, "Target values are {0, 1}", details)


def check_class_distribution(
    df: pd.DataFrame, target: str, positive_label: int, rate_min: float, rate_max: float
) -> CheckResult:
    """Report class counts and fraud rate; warn when the rate is outside the expected range."""
    counts = _class_counts(df[target])
    fraud_count = counts.get(str(positive_label), 0)
    fraud_rate = fraud_count / len(df) if len(df) else 0.0
    details = {
        "class_counts": counts,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "expected_fraud_rate_range": [rate_min, rate_max],
    }
    summary = f"{fraud_count} frauds / {len(df)} rows ({fraud_rate:.4%})"
    if not rate_min <= fraud_rate <= rate_max:
        return CheckResult(
            "class_distribution",
            Status.WARN,
            f"Fraud rate outside expected range [{rate_min}, {rate_max}]: {summary}",
            details,
        )
    return CheckResult("class_distribution", Status.PASS, summary, details)


def check_negative_values(df: pd.DataFrame, columns: Sequence[str]) -> CheckResult:
    """Fail if any of ``columns`` contains negative values."""
    per_column = {c: int((df[c] < 0).sum()) for c in columns}
    negative = {c: n for c, n in per_column.items() if n}
    details = {"per_column": per_column}
    if negative:
        return CheckResult("negative_values", Status.FAIL, f"Negative values found: {negative}", details)
    return CheckResult("negative_values", Status.PASS, f"No negative values in {list(columns)}", details)


def report_outliers(df: pd.DataFrame, columns: Sequence[str], iqr_multiplier: float) -> CheckResult:
    """Count values outside ``[Q1 - k*IQR, Q3 + k*IQR]`` per column. Report only."""
    per_column: dict[str, dict[str, float | int]] = {}
    for column in columns:
        q1, q3 = df[column].quantile([0.25, 0.75]).tolist()
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        n_outliers = int(((df[column] < lower) | (df[column] > upper)).sum())
        per_column[column] = {"lower_bound": lower, "upper_bound": upper, "count": n_outliers}

    total = sum(int(v["count"]) for v in per_column.values())
    return CheckResult(
        "outliers",
        Status.PASS,
        f"{total} outlier values reported (IQR x {iqr_multiplier}); rows are not removed",
        {"iqr_multiplier": iqr_multiplier, "per_column": per_column, "total": total},
    )


def check_duplicates(df: pd.DataFrame, target: str) -> CheckResult:
    """Count exact duplicate rows (all columns), excluding each first occurrence."""
    mask = df.duplicated(keep="first")
    count = int(mask.sum())
    details = {"count": count, "by_class": _class_counts(df.loc[mask, target])}
    if count:
        return CheckResult(
            "duplicates",
            Status.WARN,
            f"{count} exact duplicate rows will be removed (first occurrence kept)",
            details,
        )
    return CheckResult("duplicates", Status.PASS, "No duplicate rows", details)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first occurrence (DOC-02 §3 duplicate policy)."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


# ------------------------------------------------------------------------ orchestration


def _skipped(names: Iterable[str], reason: str) -> list[CheckResult]:
    return [CheckResult(name, Status.SKIPPED, f"Skipped: {reason}") for name in names]


def run_checks(
    df: pd.DataFrame, schema: SchemaConfig, validation: ValidationConfig
) -> list[CheckResult]:
    """Run all checks in dependency order, skipping those whose preconditions failed."""
    schema_check = check_schema(df, schema.required_columns)
    checks = [schema_check, check_row_count(df, validation.min_rows)]
    column_checks = [
        "dtypes",
        "missing_values",
        "infinite_values",
        "target_values",
        "class_distribution",
        "negative_values",
        "outliers",
        "duplicates",
    ]
    if schema_check.status is Status.FAIL:
        return checks + _skipped(column_checks, "required columns missing")

    numeric_columns = [c for c in schema.required_columns if c != schema.target]
    dtype_check = check_dtypes(df, schema)
    checks += [
        dtype_check,
        check_missing_values(df, schema.required_columns, schema.target),
    ]

    if dtype_check.status is Status.FAIL:
        checks += _skipped(
            ["infinite_values", "target_values", "class_distribution", "negative_values", "outliers"],
            "invalid data types",
        )
    else:
        checks.append(check_infinite_values(df, numeric_columns))
        target_check = check_target_values(df, schema.target)
        checks.append(target_check)
        if target_check.status is Status.FAIL:
            checks += _skipped(["class_distribution"], "invalid target values")
        else:
            checks.append(
                check_class_distribution(
                    df,
                    schema.target,
                    schema.positive_label,
                    validation.expected_fraud_rate_min,
                    validation.expected_fraud_rate_max,
                )
            )
        checks.append(check_negative_values(df, validation.non_negative_columns))
        checks.append(report_outliers(df, schema.features, validation.outlier_iqr_multiplier))

    checks.append(check_duplicates(df, schema.target))

    for check in checks:
        level = {Status.FAIL: logging.ERROR, Status.WARN: logging.WARNING}.get(check.status, logging.INFO)
        logger.log(level, "[%s] %s: %s", check.status.value, check.name, check.message)
    return checks


def _worst(statuses: Iterable[Status]) -> Status:
    statuses = set(statuses)
    for status in (Status.FAIL, Status.WARN, Status.PASS):
        if status in statuses:
            return status
    return Status.SKIPPED


def _details(checks: dict[str, CheckResult], name: str) -> dict[str, Any] | None:
    check = checks.get(name)
    if check is None or check.status is Status.SKIPPED:
        return None
    return check.details


def build_report(
    checks: Sequence[CheckResult],
    raw_data: pd.DataFrame | None,
    clean_data: pd.DataFrame | None,
    schema: SchemaConfig,
    source_path: str | Path | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    """Assemble the JSON-serialisable validation report.

    ``raw_data`` may be ``None`` when the file could not be loaded; dataset-level
    fields are then ``None``.
    """
    by_name = {c.name: c for c in checks}
    overall = Status.FAIL if any(c.status is Status.FAIL for c in checks) else Status.PASS
    # Schema validation = required/unexpected columns + data types; SKIPPED if never evaluated.
    schema_status = _worst(by_name[n].status for n in ("schema", "dtypes") if n in by_name)

    fraud_count = fraud_rate = class_counts = None
    if raw_data is not None and schema.target in raw_data.columns:
        distribution = _details(by_name, "class_distribution")
        if distribution is not None:
            class_counts = distribution["class_counts"]
            fraud_count = distribution["fraud_count"]
            fraud_rate = distribution["fraud_rate"]

    missing = _details(by_name, "missing_values")
    infinite = _details(by_name, "infinite_values")
    duplicates = _details(by_name, "duplicates")

    clean_summary = None
    if clean_data is not None:
        clean_counts = _class_counts(clean_data[schema.target])
        clean_fraud = clean_counts.get(str(schema.positive_label), 0)
        clean_summary = {
            "total_rows": len(clean_data),
            "total_columns": clean_data.shape[1],
            "class_counts": clean_counts,
            "fraud_count": clean_fraud,
            "fraud_rate": clean_fraud / len(clean_data) if len(clean_data) else 0.0,
            "duplicates_removed": duplicates["count"] if duplicates else 0,
        }

    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "overall_status": overall.value,
        "schema_validation_status": schema_status.value,
        "dataset": {
            "path": Path(source_path).as_posix() if source_path is not None else None,
            "sha256": sha256,
        },
        "total_rows": len(raw_data) if raw_data is not None else None,
        "total_columns": raw_data.shape[1] if raw_data is not None else None,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "class_counts": class_counts,
        "missing_values": missing["per_column"] if missing else None,
        "missing_values_total": missing["total"] if missing else None,
        "infinite_values_total": infinite["total"] if infinite else None,
        "duplicate_rows": duplicates["count"] if duplicates else None,
        "duplicate_rows_by_class": duplicates["by_class"] if duplicates else None,
        "clean_dataset": clean_summary,
        "errors": [c.message for c in checks if c.status is Status.FAIL],
        "warnings": [c.message for c in checks if c.status is Status.WARN],
        "checks": [c.to_dict() for c in checks],
    }


def validate_dataframe(
    df: pd.DataFrame,
    schema: SchemaConfig,
    validation: ValidationConfig,
    *,
    source_path: str | Path | None = None,
    sha256: str | None = None,
) -> ValidationResult:
    """Validate an in-memory dataset and, if it passes, deduplicate it.

    The clean dataset keeps only the required columns, in configured order, after
    exact duplicates (compared across all loaded columns) are removed.
    """
    checks = run_checks(df, schema, validation)
    passed = not any(c.status is Status.FAIL for c in checks)

    clean = None
    if passed:
        clean = remove_duplicates(df)[list(schema.required_columns)]
        logger.info("Clean dataset: %d rows (%d duplicates removed)", len(clean), len(df) - len(clean))

    report = build_report(checks, df, clean, schema, source_path, sha256)
    return ValidationResult(report=report, clean_data=clean)


def validate_dataset(
    path: str | Path, schema: SchemaConfig, validation: ValidationConfig
) -> ValidationResult:
    """Load the CSV at ``path``, hash it, and validate it.

    Raises:
        DataLoadError: If the file is absent, empty or unreadable.
    """
    df = load_raw_data(path)
    sha256 = compute_file_sha256(path)
    logger.info("SHA-256 of %s: %s", path, sha256)
    return validate_dataframe(df, schema, validation, source_path=path, sha256=sha256)


# --------------------------------------------------------------------------- outputs


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write the report as indented JSON, atomically (temp file then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # LF on every platform: the report is a Git-tracked DVC output (DOC-05 §23 DEV-15).
    tmp.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(tmp, path)
    logger.info("Validation report written to %s", path)
    return path


def write_clean_data(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the deduplicated dataset as CSV without an index, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    logger.info("Clean dataset written to %s (%d rows)", path, len(df))
    return path
