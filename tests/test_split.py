"""Tests for the temporal carve-out and stratified primary split (DOC-05 M3, §15)."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import load_config
from fraud_detection.data.split import (
    HOLDOUT,
    PARTITION_FILENAMES,
    PRIMARY_SPLITS,
    SplitError,
    add_row_id,
    carve_temporal_holdout,
    split_dataset,
    stratified_split,
    verify_partitions,
)
from fraud_detection.pipeline.split import main, stage_outputs

N_ROWS, N_FRAUD = 2000, 100


@pytest.fixture
def df(make_transactions) -> pd.DataFrame:
    return make_transactions(n_rows=N_ROWS, n_fraud=N_FRAUD, seed=1)


@pytest.fixture
def result(df, schema_config, split_config, temporal_config):
    return split_dataset(df, schema_config, split_config, temporal_config, seed=42)


def _ids(frame: pd.DataFrame) -> set[int]:
    return set(frame["row_id"].tolist())


# ------------------------------------------------------------------------ temporal carve-out


def test_add_row_id_is_positional_first_column(df) -> None:
    out = add_row_id(df, "row_id")
    assert out.columns[0] == "row_id"
    assert out["row_id"].tolist() == list(range(len(df)))
    assert "row_id" not in df.columns  # input untouched


def test_holdout_is_latest_fraction(df) -> None:
    carve = carve_temporal_holdout(add_row_id(df, "row_id"), "Time", 0.15)

    assert carve.nominal_cut_index == math.floor(0.85 * N_ROWS)
    assert carve.tie_rows_moved == 0  # uniform float times: no ties
    assert len(carve.holdout) == N_ROWS - math.floor(0.85 * N_ROWS)
    assert carve.pool["Time"].max() < carve.holdout["Time"].min()
    assert carve.holdout["Time"].is_monotonic_increasing


def test_cut_index_floors_despite_float_error() -> None:
    frame = pd.DataFrame({"Time": np.arange(20, dtype=float), "Class": 0})
    carve = carve_temporal_holdout(frame, "Time", 0.15)  # 0.85 * 20 -> 17
    assert carve.nominal_cut_index == 17
    assert len(carve.holdout) == 3


def test_boundary_ties_move_to_holdout() -> None:
    # 10 rows; nominal cut floor(0.85 * 10) = 8 lands inside the tie block at Time=5.
    times = [0, 1, 2, 3, 5, 5, 5, 5, 5, 9]
    frame = add_row_id(pd.DataFrame({"Time": np.array(times, dtype=float), "Class": 0}), "row_id")

    carve = carve_temporal_holdout(frame, "Time", 0.15)

    assert carve.nominal_cut_index == 8
    assert carve.cut_index == 4
    assert carve.tie_rows_moved == 4
    assert carve.boundary_time == 5.0
    assert carve.pool["Time"].max() < carve.holdout["Time"].min()
    assert (carve.holdout["Time"] == 5).sum() == 5


def test_stable_sort_keeps_file_order_for_ties() -> None:
    frame = add_row_id(
        pd.DataFrame({"Time": [3.0, 1.0, 3.0, 1.0, 3.0, 0.0, 3.0], "Class": 0}), "row_id"
    )
    carve = carve_temporal_holdout(frame, "Time", 0.5)
    tied = carve.holdout[carve.holdout["Time"] == 3.0]
    assert tied["row_id"].tolist() == [0, 2, 4, 6]


def test_unsorted_input_gives_same_holdout(df) -> None:
    with_ids = add_row_id(df, "row_id")
    shuffled = with_ids.sample(frac=1, random_state=7)  # row ids travel with the rows

    ordered_carve = carve_temporal_holdout(with_ids, "Time", 0.15)
    shuffled_carve = carve_temporal_holdout(shuffled, "Time", 0.15)

    assert _ids(shuffled_carve.holdout) == _ids(ordered_carve.holdout)
    assert _ids(ordered_carve.holdout) == set(range(N_ROWS - len(ordered_carve.holdout), N_ROWS))


def test_pool_cannot_be_empty() -> None:
    frame = pd.DataFrame({"Time": [7.0] * 10, "Class": 0})
    with pytest.raises(SplitError, match="Development pool would be empty"):
        carve_temporal_holdout(frame, "Time", 0.15)


def test_holdout_is_not_stratified(make_transactions, schema_config, split_config, temporal_config) -> None:
    frame = make_transactions(n_rows=N_ROWS, n_fraud=0)
    frame.loc[frame.index[:60], "Class"] = 1  # 60 early frauds (pool)
    frame.loc[frame.index[-40:], "Class"] = 1  # 40 late frauds (holdout)

    res = split_dataset(frame, schema_config, split_config, temporal_config, seed=42)

    holdout = res.temporal_summary["temporal_holdout"]
    assert holdout["fraud_count"] == 40  # natural prevalence kept, not rebalanced
    assert holdout["prevalence"] > 3 * res.temporal_summary["development_pool"]["prevalence"]


# --------------------------------------------------------------------------- primary split


def test_split_ratios_and_stratification(result) -> None:
    pool = result.split_summary["development_pool"]
    for name, fraction in zip(PRIMARY_SPLITS, (0.70, 0.15, 0.15), strict=True):
        stats = result.split_summary["splits"][name]
        assert abs(stats["rows"] - fraction * pool["rows"]) <= 1
        # Stratification is exact up to one fraud of rounding per split.
        assert abs(stats["fraud_count"] - fraction * pool["fraud_count"]) <= 1
        assert stats["prevalence_diff_pp"] <= 100 / stats["rows"]
        assert set(result.partitions[name]["Class"]) == {0, 1}


def test_prevalence_tolerance_flag(df, schema_config, split_config, temporal_config) -> None:
    # SC-02's 0.02 pp is only attainable on realistically sized splits (one fraud in a
    # 36k-row split ~ 0.003 pp); on 2k synthetic rows one fraud ~ 0.4 pp.
    loose = dataclasses.replace(split_config, prevalence_tolerance_pp=1.0)
    strict = dataclasses.replace(split_config, prevalence_tolerance_pp=1e-9)

    ok = split_dataset(df, schema_config, loose, temporal_config, seed=42).split_summary
    tight = split_dataset(df, schema_config, strict, temporal_config, seed=42).split_summary

    assert ok["all_within_tolerance"] is True
    assert tight["all_within_tolerance"] is False
    assert all(
        s["within_tolerance"] == (s["prevalence_diff_pp"] <= 1e-9) for s in tight["splits"].values()
    )


def test_partitions_disjoint_and_complete(result) -> None:
    ids = [_ids(result.partitions[name]) for name in (*PRIMARY_SPLITS, HOLDOUT)]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert not ids[i] & ids[j]
    assert set().union(*ids) == set(range(N_ROWS))
    assert result.split_summary["integrity"] == {
        "row_overlap": 0, "rows_assigned": N_ROWS, "all_rows_assigned": True,
    }


def test_pool_strictly_before_holdout(result) -> None:
    pool_max = max(result.partitions[n]["Time"].max() for n in PRIMARY_SPLITS)
    assert pool_max < result.partitions[HOLDOUT]["Time"].min()
    assert result.temporal_summary["ordering_invariant_holds"] is True


def test_rows_are_unchanged(df, result) -> None:
    combined = pd.concat(result.partitions.values()).sort_values("row_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(combined.drop(columns="row_id"), df)


def test_primary_partitions_sorted_by_row_id(result) -> None:
    for name in PRIMARY_SPLITS:
        assert result.partitions[name]["row_id"].is_monotonic_increasing


def test_verify_partitions_detects_overlap() -> None:
    parts = {"a": pd.DataFrame({"row_id": [0, 1]}), "b": pd.DataFrame({"row_id": [1, 2]})}
    with pytest.raises(SplitError, match="integrity"):
        verify_partitions(parts, "row_id", 3)


def test_verify_partitions_detects_unassigned_rows() -> None:
    parts = {"a": pd.DataFrame({"row_id": [0, 1]})}
    with pytest.raises(SplitError, match="1 unassigned"):
        verify_partitions(parts, "row_id", 3)


def test_split_is_reproducible(df, schema_config, split_config, temporal_config) -> None:
    first = split_dataset(df, schema_config, split_config, temporal_config, seed=42)
    second = split_dataset(df, schema_config, split_config, temporal_config, seed=42)
    for name in first.partitions:
        pd.testing.assert_frame_equal(first.partitions[name], second.partitions[name])
    assert first.split_summary == second.split_summary


def test_different_seed_changes_primary_split(df, schema_config, split_config, temporal_config) -> None:
    a = split_dataset(df, schema_config, split_config, temporal_config, seed=42)
    b = split_dataset(df, schema_config, split_config, temporal_config, seed=7)
    assert _ids(a.partitions["train"]) != _ids(b.partitions["train"])
    assert _ids(a.partitions[HOLDOUT]) == _ids(b.partitions[HOLDOUT])  # chronology, not seed


def test_too_few_frauds_to_stratify(make_transactions, schema_config, split_config, temporal_config) -> None:
    frame = make_transactions(n_rows=500, n_fraud=0)
    frame.loc[frame.index[0], "Class"] = 1
    with pytest.raises(SplitError):
        split_dataset(frame, schema_config, split_config, temporal_config, seed=42)


def test_unstratified_split_supported(df, schema_config, split_config, temporal_config) -> None:
    unstratified = dataclasses.replace(split_config, stratify=False)
    res = split_dataset(df, schema_config, unstratified, temporal_config, seed=42)
    assert res.split_summary["stratify"] is False
    assert sum(len(res.partitions[n]) for n in PRIMARY_SPLITS) == res.split_summary["development_pool"]["rows"]


def test_stratified_split_rejects_single_class_split(split_config) -> None:
    pool = pd.DataFrame({"row_id": range(20), "Class": [0] * 20})
    with pytest.raises(SplitError):
        stratified_split(pool, "Class", split_config, seed=42)


# ---------------------------------------------------------------------------- input guards


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.drop(columns=["Amount"]), "missing required columns"),
        (lambda d: d.assign(row_id=1), "reserved for row identity"),
        (lambda d: d.iloc[0:0], "no rows"),
        (lambda d: d.assign(Time=d["Time"].where(d.index != 3)), "'Time' has 1 missing"),
        (lambda d: d.assign(Class=d["Class"].replace({1: 2})), "outside"),
    ],
)
def test_invalid_input_raises(df, schema_config, split_config, temporal_config, mutate, message) -> None:
    with pytest.raises(SplitError, match=message):
        split_dataset(mutate(df), schema_config, split_config, temporal_config, seed=42)


# ------------------------------------------------------------------------------ summaries


def test_summary_counts_are_consistent(result) -> None:
    s, t = result.split_summary, result.temporal_summary
    total_rows = sum(s["splits"][n]["rows"] for n in PRIMARY_SPLITS) + s["temporal_holdout"]["rows"]
    total_fraud = sum(s["splits"][n]["fraud_count"] for n in PRIMARY_SPLITS) + s["temporal_holdout"]["fraud_count"]
    assert total_rows == s["input"]["rows"] == N_ROWS
    assert total_fraud == s["input"]["fraud_count"] == N_FRAUD
    assert t["development_pool"]["rows"] + t["temporal_holdout"]["rows"] == N_ROWS
    assert t["stratified"] is False
    assert s["feature_order"][-1] == "Amount" and "Time" not in s["feature_order"]


def test_low_confidence_flag(df, schema_config, split_config, temporal_config) -> None:
    res = split_dataset(df, schema_config, split_config, temporal_config, seed=42)
    frauds = res.temporal_summary["temporal_holdout"]["fraud_count"]
    assert res.temporal_summary["low_confidence"] is (frauds < 20)

    strict = dataclasses.replace(temporal_config, min_fraud_warning=frauds + 1)
    assert split_dataset(df, schema_config, split_config, strict, seed=42).temporal_summary["low_confidence"]


def test_summaries_are_json_serialisable(result) -> None:
    json.dumps(result.split_summary, allow_nan=False)
    json.dumps(result.temporal_summary, allow_nan=False)


# ----------------------------------------------------------------------------------- CLI


def _write_clean(config, frame: pd.DataFrame) -> None:
    config.paths.interim.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.paths.interim, index=False)


def test_stage_writes_deterministic_outputs(config_file, df) -> None:
    config = load_config(config_file)
    _write_clean(config, df)

    assert main(["--config", str(config_file)]) == 0
    first = {p: p.read_bytes() for p in stage_outputs(config)}
    assert main(["--config", str(config_file)]) == 0

    assert {p: p.read_bytes() for p in stage_outputs(config)} == first
    assert set(PARTITION_FILENAMES.values()) <= {p.name for p in first}
    train = pd.read_csv(config.paths.processed_dir / "train.csv")
    assert train.columns[0] == "row_id" and "Time" in train.columns


def test_stage_missing_input_fails_and_removes_stale_outputs(config_file, df) -> None:
    config = load_config(config_file)
    _write_clean(config, df)
    assert main(["--config", str(config_file)]) == 0

    config.paths.interim.unlink()
    assert main(["--config", str(config_file)]) == 1
    assert not any(p.exists() for p in stage_outputs(config))


def test_stage_invalid_config_exit_code(tmp_path) -> None:
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2
