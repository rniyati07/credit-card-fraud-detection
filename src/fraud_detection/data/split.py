"""Temporal carve-out and stratified primary split (DOC-02 §6, §21.3; DOC-05 M3).

Order of operations (leakage-safe, nothing is fitted here):

1. Attach a stable ``row_id`` (position in the clean file) for integrity checks
   (DOC-05 IMP-03). It is carried in every partition but is never a model feature.
2. Stable-sort by the ordering column (``kind="mergesort"``; ties keep file order),
   cut at ``floor((1 - holdout_fraction) * n)`` and move rows sharing the boundary
   time into the holdout, so the pool is strictly earlier than the holdout.
   The holdout is **not** stratified; it keeps its natural fraud prevalence.
3. Split the development pool with two seeded ``train_test_split`` calls:
   train vs. temp (``val + test``), then temp into val and test, stratified on
   the target.
4. Verify that the four partitions are disjoint and cover every row.

Primary partitions are written sorted by ``row_id``; the holdout is written in
chronological order. Both orders are deterministic.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import SchemaConfig, SplitConfig, TemporalConfig

logger = logging.getLogger(__name__)

PRIMARY_SPLITS = ("train", "val", "test")
HOLDOUT = "temporal_holdout"
PARTITION_FILENAMES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
    HOLDOUT: "temporal_holdout.csv",
}


class SplitError(Exception):
    """Raised when the input cannot be split safely (bad input, empty or degenerate partitions)."""


@dataclass(frozen=True)
class TemporalCarveOut:
    """Development pool and temporal holdout, plus how the cut was placed."""

    pool: pd.DataFrame
    holdout: pd.DataFrame
    nominal_cut_index: int
    cut_index: int
    boundary_time: float

    @property
    def tie_rows_moved(self) -> int:
        """Rows moved from the pool into the holdout because they share the boundary time."""
        return self.nominal_cut_index - self.cut_index


@dataclass(frozen=True)
class SplitResult:
    """All four partitions and their summaries."""

    partitions: dict[str, pd.DataFrame]
    split_summary: dict[str, Any]
    temporal_summary: dict[str, Any]


# ------------------------------------------------------------------------------ input


def load_clean_data(path: str | Path) -> pd.DataFrame:
    """Load the validated, deduplicated dataset produced by the ``validate`` stage.

    Raises:
        SplitError: If the file is missing or unreadable.
    """
    path = Path(path)
    if not path.is_file():
        raise SplitError(
            f"Clean dataset not found at '{path}'. "
            "Run `python -m fraud_detection.pipeline.validate` first."
        )
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        raise SplitError(f"Clean dataset at '{path}' could not be read: {exc}") from exc
    logger.info("Loaded %s: %d rows x %d columns", path, df.shape[0], df.shape[1])
    return df


def check_split_input(df: pd.DataFrame, schema: SchemaConfig, row_id_column: str) -> None:
    """Guard the preconditions the split relies on (M1 should already guarantee them).

    Raises:
        SplitError: On missing columns, a clashing row-id column, no rows, missing
            ordering/target values, or target values outside {0, 1}.
    """
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise SplitError(f"Input is missing required columns: {missing}")
    if row_id_column in df.columns:
        raise SplitError(f"Input already has a '{row_id_column}' column (reserved for row identity)")
    if df.empty:
        raise SplitError("Input dataset has no rows")
    for column in (schema.ordering_column, schema.target):
        n_missing = int(df[column].isna().sum())
        if n_missing:
            raise SplitError(f"Column '{column}' has {n_missing} missing values; cannot split")
    invalid = sorted(set(df[schema.target].unique()) - {0, 1})
    if invalid:
        raise SplitError(f"Target '{schema.target}' has values outside {{0, 1}}: {invalid}")


def add_row_id(df: pd.DataFrame, row_id_column: str) -> pd.DataFrame:
    """Return a copy with ``row_id`` (0..n-1, position in the input) as the first column."""
    out = df.reset_index(drop=True)
    out.insert(0, row_id_column, np.arange(len(out), dtype=np.int64))
    return out


# ---------------------------------------------------------------------------- splitting


def carve_temporal_holdout(
    df: pd.DataFrame, ordering_column: str, holdout_fraction: float
) -> TemporalCarveOut:
    """Reserve the chronologically latest rows as the temporal holdout (DOC-02 §21.3).

    Raises:
        SplitError: If the holdout or the development pool would be empty.
    """
    ordered = df.sort_values(ordering_column, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    # Round first so that e.g. 0.85 * 20 = 16.999999... still floors to 17.
    nominal = math.floor(round((1.0 - holdout_fraction) * n, 9))
    if nominal >= n:
        raise SplitError(f"Temporal holdout would be empty ({n} rows, fraction {holdout_fraction})")

    times = ordered[ordering_column].to_numpy()
    boundary_time = float(times[nominal])
    cut = int(np.searchsorted(times, boundary_time, side="left"))  # ties -> holdout
    if cut == 0:
        raise SplitError(
            "Development pool would be empty: every row shares the boundary "
            f"{ordering_column} value {boundary_time}"
        )

    carve = TemporalCarveOut(
        pool=ordered.iloc[:cut].reset_index(drop=True),
        holdout=ordered.iloc[cut:].reset_index(drop=True),
        nominal_cut_index=nominal,
        cut_index=cut,
        boundary_time=boundary_time,
    )
    logger.info(
        "Temporal carve-out: pool %d rows, holdout %d rows (boundary %s=%s, %d tie rows moved)",
        len(carve.pool), len(carve.holdout), ordering_column, boundary_time, carve.tie_rows_moved,
    )
    return carve


def stratified_split(
    pool: pd.DataFrame, target: str, split: SplitConfig, seed: int
) -> dict[str, pd.DataFrame]:
    """Split the pool into train/val/test with two seeded (stratified) ``train_test_split`` calls.

    Each partition is returned sorted by ``split.row_id_column``.

    Raises:
        SplitError: If stratification is impossible or a split would lack a class.
    """
    positions = np.arange(len(pool))
    y = pool[target].to_numpy()
    try:
        train_idx, temp_idx = train_test_split(
            positions,
            test_size=split.val + split.test,
            stratify=y if split.stratify else None,
            random_state=seed,
            shuffle=True,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=split.test / (split.val + split.test),
            stratify=y[temp_idx] if split.stratify else None,
            random_state=seed,
            shuffle=True,
        )
    except ValueError as exc:
        raise SplitError(f"Cannot split the development pool: {exc}") from exc

    partitions = {}
    for name, idx in zip(PRIMARY_SPLITS, (train_idx, val_idx, test_idx), strict=True):
        part = pool.iloc[idx].sort_values(split.row_id_column, kind="mergesort")
        classes = set(part[target].unique())
        if classes != {0, 1}:
            raise SplitError(f"Split '{name}' contains only class(es) {sorted(classes)}")
        partitions[name] = part.reset_index(drop=True)
    return partitions


def verify_partitions(
    partitions: dict[str, pd.DataFrame], row_id_column: str, n_rows: int
) -> dict[str, Any]:
    """Assert the partitions are pairwise disjoint and together cover all ``n_rows`` rows.

    Raises:
        SplitError: If any row is shared, duplicated or unassigned.
    """
    ids = {name: part[row_id_column].to_numpy() for name, part in partitions.items()}
    all_ids = np.concatenate(list(ids.values()))
    overlap = len(all_ids) - len(np.unique(all_ids))
    unassigned = n_rows - len(np.unique(all_ids))
    if overlap or unassigned:
        raise SplitError(
            f"Partition integrity violated: {overlap} shared/duplicated rows, {unassigned} unassigned"
        )
    return {"row_overlap": 0, "rows_assigned": int(len(all_ids)), "all_rows_assigned": True}


# ---------------------------------------------------------------------------- summaries


def partition_stats(df: pd.DataFrame, target: str, positive_label: int) -> dict[str, Any]:
    """Row count, class counts and prevalence of one partition (FR-006)."""
    rows = int(len(df))
    fraud = int((df[target] == positive_label).sum())
    return {
        "rows": rows,
        "fraud_count": fraud,
        "legitimate_count": rows - fraud,
        "prevalence": fraud / rows if rows else 0.0,
    }


def _time_range(df: pd.DataFrame, ordering_column: str) -> dict[str, float]:
    return {
        f"{ordering_column.lower()}_min": float(df[ordering_column].min()),
        f"{ordering_column.lower()}_max": float(df[ordering_column].max()),
    }


def build_temporal_summary(
    carve: TemporalCarveOut, schema: SchemaConfig, temporal: TemporalConfig, n_rows: int
) -> dict[str, Any]:
    """Carve-out metadata for the later temporal robustness analysis (DOC-02 §21.3 rule 5)."""
    target, pos, order = schema.target, schema.positive_label, schema.ordering_column
    pool = partition_stats(carve.pool, target, pos) | _time_range(carve.pool, order)
    holdout = partition_stats(carve.holdout, target, pos) | _time_range(carve.holdout, order)
    holdout["share_of_total"] = holdout["rows"] / n_rows
    return {
        "ordering_column": order,
        "sort": "ascending, stable (mergesort); ties keep file order",
        "holdout_fraction": temporal.holdout_fraction,
        "stratified": False,
        "total_rows": n_rows,
        "nominal_cut_index": carve.nominal_cut_index,
        "cut_index": carve.cut_index,
        "boundary_time": carve.boundary_time,
        "boundary_tie_rows_moved_to_holdout": carve.tie_rows_moved,
        "development_pool": pool,
        "temporal_holdout": holdout,
        "ordering_invariant_holds": pool[f"{order.lower()}_max"] < holdout[f"{order.lower()}_min"],
        "holdout_to_pool_prevalence_ratio": (
            holdout["prevalence"] / pool["prevalence"] if pool["prevalence"] else None
        ),
        "min_fraud_warning": temporal.min_fraud_warning,
        "low_confidence": holdout["fraud_count"] < temporal.min_fraud_warning,
    }


def build_split_summary(
    df: pd.DataFrame,
    carve: TemporalCarveOut,
    partitions: dict[str, pd.DataFrame],
    integrity: dict[str, Any],
    schema: SchemaConfig,
    split: SplitConfig,
    seed: int,
    source_path: str | Path | None,
    sha256: str | None,
) -> dict[str, Any]:
    """Counts, prevalence and stratification tolerance for every partition (FR-006, SC-02)."""
    target, pos = schema.target, schema.positive_label
    pool = partition_stats(carve.pool, target, pos)
    splits = {}
    for name in PRIMARY_SPLITS:
        stats = partition_stats(partitions[name], target, pos)
        diff_pp = abs(stats["prevalence"] - pool["prevalence"]) * 100
        splits[name] = stats | {
            "share_of_pool": stats["rows"] / pool["rows"],
            "prevalence_diff_pp": diff_pp,
            "within_tolerance": diff_pp <= split.prevalence_tolerance_pp,
        }
    holdout = partition_stats(partitions[HOLDOUT], target, pos)
    holdout["share_of_total"] = holdout["rows"] / len(df)

    return {
        "seed": seed,
        "input": {
            "path": Path(source_path).as_posix() if source_path is not None else None,
            "sha256": sha256,
        }
        | partition_stats(df, target, pos),
        "target": target,
        "ordering_column": schema.ordering_column,
        "row_id_column": split.row_id_column,
        "feature_order": list(schema.features),
        "fractions": {"train": split.train, "val": split.val, "test": split.test},
        "stratify": split.stratify,
        "prevalence_tolerance_pp": split.prevalence_tolerance_pp,
        "development_pool": pool,
        "splits": splits,
        "temporal_holdout": holdout,
        "all_within_tolerance": all(s["within_tolerance"] for s in splits.values()),
        "integrity": integrity,
        "files": dict(PARTITION_FILENAMES),
    }


# --------------------------------------------------------------------------- orchestration


def split_dataset(
    df: pd.DataFrame,
    schema: SchemaConfig,
    split: SplitConfig,
    temporal: TemporalConfig,
    seed: int,
    *,
    source_path: str | Path | None = None,
    sha256: str | None = None,
) -> SplitResult:
    """Carve out the temporal holdout, split the pool, verify integrity and summarise.

    Raises:
        SplitError: If the input is invalid or any partition would be degenerate.
    """
    check_split_input(df, schema, split.row_id_column)
    data = add_row_id(df, split.row_id_column)

    carve = carve_temporal_holdout(data, schema.ordering_column, temporal.holdout_fraction)
    partitions = stratified_split(carve.pool, schema.target, split, seed)
    partitions[HOLDOUT] = carve.holdout
    integrity = verify_partitions(partitions, split.row_id_column, len(data))

    split_summary = build_split_summary(
        data, carve, partitions, integrity, schema, split, seed, source_path, sha256
    )
    temporal_summary = build_temporal_summary(carve, schema, temporal, len(data))

    for name in PRIMARY_SPLITS:
        s = split_summary["splits"][name]
        logger.info(
            "%-5s %7d rows, %4d frauds (prevalence %.4f%%, diff %.4f pp)",
            name, s["rows"], s["fraud_count"], 100 * s["prevalence"], s["prevalence_diff_pp"],
        )
        if not s["within_tolerance"]:
            logger.warning(
                "Split '%s' prevalence differs from the pool by %.4f pp (> %.4f pp tolerance)",
                name, s["prevalence_diff_pp"], split.prevalence_tolerance_pp,
            )
    h = temporal_summary["temporal_holdout"]
    logger.info(
        "holdout %5d rows, %4d frauds (prevalence %.4f%%)", h["rows"], h["fraud_count"], 100 * h["prevalence"]
    )
    if temporal_summary["low_confidence"]:
        logger.warning(
            "Temporal holdout has %d frauds (< %d): its metrics will be low-confidence",
            h["fraud_count"], temporal.min_fraud_warning,
        )
    return SplitResult(partitions, split_summary, temporal_summary)


# ------------------------------------------------------------------------------ outputs


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write indented JSON atomically (temp file then rename). No timestamps: deterministic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    logger.info("Wrote %s", path)
    return path


def write_partitions(partitions: dict[str, pd.DataFrame], processed_dir: str | Path) -> dict[str, Path]:
    """Write each partition as CSV (no index) atomically into ``processed_dir``."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, filename in PARTITION_FILENAMES.items():
        path = processed_dir / filename
        tmp = path.with_name(path.name + ".tmp")
        partitions[name].to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, path)
        logger.info("Wrote %s (%d rows)", path, len(partitions[name]))
        paths[name] = path
    return paths
