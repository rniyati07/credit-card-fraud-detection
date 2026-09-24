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
# Invariant: exactly these three candidates may be trained (DOC-01 §9, DOC-05 §7).
CANDIDATE_MODELS = ("logistic_regression", "random_forest", "xgboost")
SUPPORTED_IMBALANCE_STRATEGIES = ("class_weight",)
# Tuning invariants (DOC-02 §10): PR-AUC scoring, at most 10 sampled candidates per model,
# and no tuning of values fixed by the imbalance strategy or the global seed.
TUNING_SCORING = "average_precision"
MAX_TUNING_ITERATIONS = 10
RESERVED_TUNING_PARAMS = ("random_state", "scale_pos_weight", "class_weight")
SUPPORTED_THRESHOLD_FALLBACKS = ("f2",)


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ProjectConfig:
    """Project-wide settings."""

    name: str
    model_version: str
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
    baseline_dir: Path
    candidates_dir: Path
    tuning_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path


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
class ImbalanceConfig:
    """Class-imbalance handling (DOC-02 §8). Only cost-sensitive learning is supported."""

    strategy: str


@dataclass(frozen=True)
class ModelsConfig:
    """Enabled candidate models and their fixed parameters (DOC-02 §9)."""

    enabled: tuple[str, ...]
    params: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class TuningConfig:
    """Train-only cross-validated hyperparameter search (DOC-02 §10)."""

    cv_folds: int
    n_iter: int
    scoring: str
    search_spaces: dict[str, dict[str, tuple[Any, ...]]]


@dataclass(frozen=True)
class ThresholdConfig:
    """Validation-only threshold optimisation (DOC-02 §13)."""

    r_min: float
    r_min_confirmed: bool
    grid_start: float
    grid_stop: float
    grid_step: float
    include_pr_curve_points: bool
    fallback: str
    reference_threshold: float


@dataclass(frozen=True)
class SelectionConfig:
    """Model-selection framework settings (DOC-02 §12)."""

    pr_auc_tie_tolerance: float


@dataclass(frozen=True)
class MlflowConfig:
    """Local MLflow tracking (DOC-03 §6)."""

    tracking_uri: str
    experiment_name: str


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
    imbalance: ImbalanceConfig
    models: ModelsConfig
    tuning: TuningConfig
    threshold: ThresholdConfig
    selection: SelectionConfig
    mlflow: MlflowConfig


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


def _non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Config key '{name}' must be a non-empty string")
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


def _parse_imbalance(section: Mapping[str, Any]) -> ImbalanceConfig:
    strategy = str(_get(section, "strategy", "imbalance"))
    if strategy not in SUPPORTED_IMBALANCE_STRATEGIES:
        raise ConfigError(
            f"imbalance.strategy must be one of {list(SUPPORTED_IMBALANCE_STRATEGIES)}, got '{strategy}'"
        )
    return ImbalanceConfig(strategy=strategy)


def _parse_models(section: Mapping[str, Any], imbalance: ImbalanceConfig) -> ModelsConfig:
    enabled = _str_list(_get(section, "enabled", "models"), "models.enabled")
    if not enabled:
        raise ConfigError("models.enabled must list at least one model")
    unknown = sorted(set(enabled) - set(CANDIDATE_MODELS))
    if unknown:
        raise ConfigError(
            f"models.enabled has unknown models {unknown}; only {list(CANDIDATE_MODELS)} are allowed"
        )

    params: dict[str, dict[str, Any]] = {}
    for name in enabled:
        model_params = section.get(name, {})
        if not isinstance(model_params, Mapping):
            raise ConfigError(f"Config key 'models.{name}' must be a mapping")
        # Injected by code: the global seed (NFR-002) and the train-only XGBoost weight (L-06).
        for reserved in ("random_state", "scale_pos_weight"):
            if reserved in model_params:
                raise ConfigError(f"models.{name}.{reserved} is set in code and must not be configured")
        params[name] = dict(model_params)

    if imbalance.strategy == "class_weight":
        for name in ("logistic_regression", "random_forest"):
            if name in params and "class_weight" not in params[name]:
                raise ConfigError(f"models.{name}.class_weight is required for the class_weight strategy")
    return ModelsConfig(enabled=enabled, params=params)


def _parse_search_space(name: str, raw: Any, fixed: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigError(f"tuning.search_spaces.{name} must be a non-empty mapping")
    space: dict[str, tuple[Any, ...]] = {}
    for param, values in raw.items():
        key = f"tuning.search_spaces.{name}.{param}"
        if param in RESERVED_TUNING_PARAMS:
            raise ConfigError(f"{key}: '{param}' is fixed by the imbalance strategy or seed and cannot be tuned")
        if param in fixed:
            raise ConfigError(f"{key}: '{param}' is already fixed in models.{name}")
        if not isinstance(values, list) or not values:
            raise ConfigError(f"{key} must be a non-empty list")
        if not all(v is None or isinstance(v, (bool, int, float, str)) for v in values):
            raise ConfigError(f"{key} values must be scalars (number, string, bool or null)")
        if len(set(map(repr, values))) != len(values):
            raise ConfigError(f"{key} contains duplicate values")
        space[str(param)] = tuple(values)
    return space


def _parse_tuning(section: Mapping[str, Any], models: ModelsConfig) -> TuningConfig:
    tuning_config = TuningConfig(
        cv_folds=_int(_get(section, "cv_folds", "tuning"), "tuning.cv_folds"),
        n_iter=_int(_get(section, "n_iter", "tuning"), "tuning.n_iter"),
        scoring=str(_get(section, "scoring", "tuning")),
        search_spaces={},
    )
    if tuning_config.cv_folds < 2:
        raise ConfigError("tuning.cv_folds must be >= 2")
    if not 1 <= tuning_config.n_iter <= MAX_TUNING_ITERATIONS:
        raise ConfigError(f"tuning.n_iter must be in [1, {MAX_TUNING_ITERATIONS}] (DOC-02 §10)")
    if tuning_config.scoring != TUNING_SCORING:
        raise ConfigError(f"tuning.scoring must be '{TUNING_SCORING}' (PR-AUC, DOC-02 §10)")

    spaces = _get(section, "search_spaces", "tuning")
    if not isinstance(spaces, Mapping):
        raise ConfigError("Config key 'tuning.search_spaces' must be a mapping")
    if set(spaces) != set(models.enabled):
        raise ConfigError(
            "tuning.search_spaces must define exactly the enabled models "
            f"(missing: {sorted(set(models.enabled) - set(spaces))}, extra: {sorted(set(spaces) - set(models.enabled))})"
        )
    for name in models.enabled:
        tuning_config.search_spaces[name] = _parse_search_space(name, spaces[name], models.params[name])
    return tuning_config


def _parse_threshold(section: Mapping[str, Any]) -> ThresholdConfig:
    grid = _get(section, "grid", "threshold")
    if not isinstance(grid, Mapping):
        raise ConfigError("Config key 'threshold.grid' must be a mapping")
    include = _get(section, "include_pr_curve_points", "threshold")
    if not isinstance(include, bool):
        raise ConfigError("Config key 'threshold.include_pr_curve_points' must be a boolean")
    confirmed = _get(section, "r_min_confirmed", "threshold")
    if not isinstance(confirmed, bool):
        raise ConfigError("Config key 'threshold.r_min_confirmed' must be a boolean")
    threshold = ThresholdConfig(
        r_min=_float(_get(section, "r_min", "threshold"), "threshold.r_min"),
        r_min_confirmed=confirmed,
        grid_start=_float(_get(grid, "start", "threshold.grid"), "threshold.grid.start"),
        grid_stop=_float(_get(grid, "stop", "threshold.grid"), "threshold.grid.stop"),
        grid_step=_float(_get(grid, "step", "threshold.grid"), "threshold.grid.step"),
        include_pr_curve_points=include,
        fallback=str(_get(section, "fallback", "threshold")),
        reference_threshold=_float(
            _get(section, "reference_threshold", "threshold"), "threshold.reference_threshold"
        ),
    )
    if not 0.0 < threshold.r_min <= 1.0:
        raise ConfigError("threshold.r_min must be in (0, 1]")
    if not 0.0 < threshold.grid_start <= threshold.grid_stop < 1.0:
        raise ConfigError("threshold.grid must satisfy 0 < start <= stop < 1")
    if threshold.grid_step <= 0:
        raise ConfigError("threshold.grid.step must be > 0")
    if threshold.fallback not in SUPPORTED_THRESHOLD_FALLBACKS:
        raise ConfigError(f"threshold.fallback must be one of {list(SUPPORTED_THRESHOLD_FALLBACKS)}")
    if not 0.0 < threshold.reference_threshold < 1.0:
        raise ConfigError("threshold.reference_threshold must be in (0, 1)")
    return threshold


def _parse_selection(section: Mapping[str, Any]) -> SelectionConfig:
    selection = SelectionConfig(
        pr_auc_tie_tolerance=_float(
            _get(section, "pr_auc_tie_tolerance", "selection"), "selection.pr_auc_tie_tolerance"
        )
    )
    if selection.pr_auc_tie_tolerance < 0:
        raise ConfigError("selection.pr_auc_tie_tolerance must be >= 0")
    return selection


def _parse_mlflow(section: Mapping[str, Any]) -> MlflowConfig:
    mlflow = MlflowConfig(
        tracking_uri=str(_get(section, "tracking_uri", "mlflow")),
        experiment_name=str(_get(section, "experiment_name", "mlflow")),
    )
    if not mlflow.tracking_uri or not mlflow.experiment_name:
        raise ConfigError("mlflow.tracking_uri and mlflow.experiment_name must be non-empty")
    return mlflow


def parse_config(raw: Mapping[str, Any]) -> Config:
    """Build a validated :class:`Config` from an already-parsed YAML mapping.

    Raises:
        ConfigError: If a section or key is missing or an invariant is violated.
    """
    project = _section(raw, "project")
    paths = _section(raw, "paths")
    schema = _parse_schema(_section(raw, "schema"))
    imbalance = _parse_imbalance(_section(raw, "imbalance"))
    models = _parse_models(_section(raw, "models"), imbalance)

    return Config(
        project=ProjectConfig(
            name=str(_get(project, "name", "project")),
            model_version=_non_empty_str(_get(project, "model_version", "project"), "project.model_version"),
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
            baseline_dir=Path(_get(paths, "baseline_dir", "paths")),
            candidates_dir=Path(_get(paths, "candidates_dir", "paths")),
            tuning_dir=Path(_get(paths, "tuning_dir", "paths")),
            models_dir=Path(_get(paths, "models_dir", "paths")),
            reports_dir=Path(_get(paths, "reports_dir", "paths")),
            figures_dir=Path(_get(paths, "figures_dir", "paths")),
        ),
        schema=schema,
        validation=_parse_validation(_section(raw, "validation"), schema),
        eda=_parse_eda(_section(raw, "eda"), schema),
        split=_parse_split(_section(raw, "split"), schema),
        temporal=_parse_temporal(_section(raw, "temporal")),
        preprocessing=_parse_preprocessing(_section(raw, "preprocessing"), schema),
        imbalance=imbalance,
        models=models,
        tuning=_parse_tuning(_section(raw, "tuning"), models),
        threshold=_parse_threshold(_section(raw, "threshold")),
        selection=_parse_selection(_section(raw, "selection")),
        mlflow=_parse_mlflow(_section(raw, "mlflow")),
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
