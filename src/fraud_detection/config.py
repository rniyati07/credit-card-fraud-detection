"""Load and validate the project configuration (``config/config.yaml``).

The YAML file is parsed into frozen dataclasses so that downstream code works with
typed, already-validated values. Invariants that must never be configurable
(DOC-05 §7) are asserted here: exactly 29 features, and neither the ordering column
(``Time``) nor the target (``Class``) may appear in the feature list.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
EXPECTED_N_FEATURES = 29


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ProjectConfig:
    """Project-wide settings."""

    name: str
    seed: int


@dataclass(frozen=True)
class PathsConfig:
    """Input/output locations, relative to the working directory (repository root)."""

    raw: Path
    interim: Path
    validation_report: Path
    eda_dir: Path
    processed_dir: Path
    split_summary: Path
    temporal_split_summary: Path


@dataclass(frozen=True)
class SchemaConfig:
    """Expected dataset schema and column roles."""

    target: str
    positive_label: int
    ordering_column: str
    features: tuple[str, ...]
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds used by the data-validation checks."""

    min_rows: int
    expected_fraud_rate_min: float
    expected_fraud_rate_max: float
    non_negative_columns: tuple[str, ...]
    outlier_iqr_multiplier: float


@dataclass(frozen=True)
class EdaConfig:
    """Presentation settings for the descriptive EDA (DOC-02 §4)."""

    top_k_features: int
    time_bin_seconds: int
    histogram_bins: int
    correlation_top_pairs: int
    figure_dpi: int


@dataclass(frozen=True)
class SplitConfig:
    """Stratified train/validation/test split of the development pool (DOC-02 §6)."""

    train: float
    val: float
    test: float
    stratify: bool
    prevalence_tolerance_pp: float
    row_id_column: str


@dataclass(frozen=True)
class TemporalConfig:
    """Chronological holdout carved out before the primary split (DOC-02 §21.3)."""

    holdout_fraction: float
    min_fraud_warning: int


@dataclass(frozen=True)
class PreprocessingConfig:
    """Feature groups of the shared ColumnTransformer (DOC-02 §7)."""

    robust_scaled_features: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    """Validated project configuration."""

    project: ProjectConfig
    paths: PathsConfig
    schema: SchemaConfig
    validation: ValidationConfig
    eda: EdaConfig
    split: SplitConfig
    temporal: TemporalConfig
    preprocessing: PreprocessingConfig


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Config section '{key}' is missing or is not a mapping")
    return value


def _get(section: Mapping[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ConfigError(f"Config key '{section_name}.{key}' is missing")
    return section[key]


def _str_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"Config key '{name}' must be a list of strings")
    if len(set(value)) != len(value):
        raise ConfigError(f"Config key '{name}' contains duplicate entries")
    return tuple(value)


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"Config key '{name}' must be an integer")
    return value


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Config key '{name}' must be a number")
    return float(value)


def _parse_schema(section: Mapping[str, Any]) -> SchemaConfig:
    schema = SchemaConfig(
        target=str(_get(section, "target", "schema")),
        positive_label=_int(_get(section, "positive_label", "schema"), "schema.positive_label"),
        ordering_column=str(_get(section, "ordering_column", "schema")),
        features=_str_list(_get(section, "features", "schema"), "schema.features"),
        required_columns=_str_list(
            _get(section, "required_columns", "schema"), "schema.required_columns"
        ),
    )

    if len(schema.features) != EXPECTED_N_FEATURES:
        raise ConfigError(
            f"schema.features must contain exactly {EXPECTED_N_FEATURES} features, "
            f"got {len(schema.features)}"
        )
    if schema.ordering_column in schema.features:
        raise ConfigError(
            f"Ordering column '{schema.ordering_column}' must not be a model feature"
        )
    if schema.target in schema.features:
        raise ConfigError(f"Target '{schema.target}' must not be a model feature")
    if schema.positive_label not in (0, 1):
        raise ConfigError("schema.positive_label must be 0 or 1")

    expected = {*schema.features, schema.target, schema.ordering_column}
    if set(schema.required_columns) != expected:
        missing = sorted(expected - set(schema.required_columns))
        extra = sorted(set(schema.required_columns) - expected)
        raise ConfigError(
            "schema.required_columns must be exactly the features plus the target and "
            f"ordering column (missing: {missing}, unexpected: {extra})"
        )
    return schema


def _parse_validation(section: Mapping[str, Any], schema: SchemaConfig) -> ValidationConfig:
    rate = _get(section, "expected_fraud_rate", "validation")
    if not isinstance(rate, Mapping):
        raise ConfigError("Config key 'validation.expected_fraud_rate' must be a mapping")

    validation = ValidationConfig(
        min_rows=_int(_get(section, "min_rows", "validation"), "validation.min_rows"),
        expected_fraud_rate_min=_float(
            _get(rate, "min", "validation.expected_fraud_rate"),
            "validation.expected_fraud_rate.min",
        ),
        expected_fraud_rate_max=_float(
            _get(rate, "max", "validation.expected_fraud_rate"),
            "validation.expected_fraud_rate.max",
        ),
        non_negative_columns=_str_list(
            _get(section, "non_negative_columns", "validation"),
            "validation.non_negative_columns",
        ),
        outlier_iqr_multiplier=_float(
            _get(section, "outlier_iqr_multiplier", "validation"),
            "validation.outlier_iqr_multiplier",
        ),
    )

    if validation.min_rows < 1:
        raise ConfigError("validation.min_rows must be >= 1")
    if not 0.0 <= validation.expected_fraud_rate_min < validation.expected_fraud_rate_max <= 1.0:
        raise ConfigError("validation.expected_fraud_rate must satisfy 0 <= min < max <= 1")
    unknown = sorted(set(validation.non_negative_columns) - set(schema.required_columns))
    if unknown:
        raise ConfigError(f"validation.non_negative_columns not in schema: {unknown}")
    if validation.outlier_iqr_multiplier <= 0:
        raise ConfigError("validation.outlier_iqr_multiplier must be > 0")
    return validation


def _parse_eda(section: Mapping[str, Any], schema: SchemaConfig) -> EdaConfig:
    eda = EdaConfig(
        **{
            key: _int(_get(section, key, "eda"), f"eda.{key}")
            for key in (
                "top_k_features",
                "time_bin_seconds",
                "histogram_bins",
                "correlation_top_pairs",
                "figure_dpi",
            )
        }
    )
    for key, value in vars(eda).items():
        if value < 1:
            raise ConfigError(f"eda.{key} must be >= 1")
    if eda.top_k_features > len(schema.features):
        raise ConfigError("eda.top_k_features cannot exceed the number of features")
    return eda


def _parse_split(section: Mapping[str, Any], schema: SchemaConfig) -> SplitConfig:
    stratify = _get(section, "stratify", "split")
    if not isinstance(stratify, bool):
        raise ConfigError("Config key 'split.stratify' must be a boolean")
    split = SplitConfig(
        train=_float(_get(section, "train", "split"), "split.train"),
        val=_float(_get(section, "val", "split"), "split.val"),
        test=_float(_get(section, "test", "split"), "split.test"),
        stratify=stratify,
        prevalence_tolerance_pp=_float(
            _get(section, "prevalence_tolerance_pp", "split"), "split.prevalence_tolerance_pp"
        ),
        row_id_column=str(_get(section, "row_id_column", "split")),
    )

    for key in ("train", "val", "test"):
        if not 0.0 < getattr(split, key) < 1.0:
            raise ConfigError(f"split.{key} must be in (0, 1)")
    if not math.isclose(split.train + split.val + split.test, 1.0, abs_tol=1e-9):
        raise ConfigError("split.train + split.val + split.test must sum to 1")
    if split.prevalence_tolerance_pp <= 0:
        raise ConfigError("split.prevalence_tolerance_pp must be > 0")
    if not split.row_id_column or split.row_id_column in schema.required_columns:
        raise ConfigError("split.row_id_column must be a non-empty name not used by the schema")
    return split


def _parse_temporal(section: Mapping[str, Any]) -> TemporalConfig:
    temporal = TemporalConfig(
        holdout_fraction=_float(
            _get(section, "holdout_fraction", "temporal"), "temporal.holdout_fraction"
        ),
        min_fraud_warning=_int(
            _get(section, "min_fraud_warning", "temporal"), "temporal.min_fraud_warning"
        ),
    )
    if not 0.0 < temporal.holdout_fraction < 1.0:
        raise ConfigError("temporal.holdout_fraction must be in (0, 1)")
    if temporal.min_fraud_warning < 0:
        raise ConfigError("temporal.min_fraud_warning must be >= 0")
    return temporal


def _parse_preprocessing(section: Mapping[str, Any], schema: SchemaConfig) -> PreprocessingConfig:
    robust = _str_list(
        _get(section, "robust_scaled_features", "preprocessing"),
        "preprocessing.robust_scaled_features",
    )
    unknown = sorted(set(robust) - set(schema.features))
    if unknown:
        raise ConfigError(f"preprocessing.robust_scaled_features not in schema.features: {unknown}")
    return PreprocessingConfig(robust_scaled_features=robust)


def parse_config(raw: Mapping[str, Any]) -> Config:
    """Build a validated :class:`Config` from an already-parsed YAML mapping.

    Raises:
        ConfigError: If a section or key is missing or an invariant is violated.
    """
    project = _section(raw, "project")
    paths = _section(raw, "paths")
    schema = _parse_schema(_section(raw, "schema"))

    return Config(
        project=ProjectConfig(
            name=str(_get(project, "name", "project")),
            seed=_int(_get(project, "seed", "project"), "project.seed"),
        ),
        paths=PathsConfig(
            raw=Path(_get(paths, "raw", "paths")),
            interim=Path(_get(paths, "interim", "paths")),
            validation_report=Path(_get(paths, "validation_report", "paths")),
            eda_dir=Path(_get(paths, "eda_dir", "paths")),
            processed_dir=Path(_get(paths, "processed_dir", "paths")),
            split_summary=Path(_get(paths, "split_summary", "paths")),
            temporal_split_summary=Path(_get(paths, "temporal_split_summary", "paths")),
        ),
        schema=schema,
        validation=_parse_validation(_section(raw, "validation"), schema),
        eda=_parse_eda(_section(raw, "eda"), schema),
        split=_parse_split(_section(raw, "split"), schema),
        temporal=_parse_temporal(_section(raw, "temporal")),
        preprocessing=_parse_preprocessing(_section(raw, "preprocessing"), schema),
    )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read and validate the YAML configuration file.

    Args:
        path: Location of ``config.yaml``.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is absent, is not valid YAML, or fails validation.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found at '{path}'")
    try:
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file '{path}' is not valid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Config file '{path}' must contain a mapping at the top level")

    config = parse_config(raw)
    logger.debug("Loaded configuration from %s", path)
    return config
