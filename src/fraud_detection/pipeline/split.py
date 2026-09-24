"""DVC stage entrypoint: ``split`` (DOC-03 §4.2, stage 2).

Reads ``paths.interim``; writes ``train/val/test/temporal_holdout.csv`` into
``paths.processed_dir``, plus ``paths.split_summary`` and ``paths.temporal_split_summary``.
All outputs are deterministic for a given input, config and seed.

Usage::

    python -m fraud_detection.pipeline.split [--config config/config.yaml]

Exit codes: 0 = success, 1 = split failed (missing/invalid input or degenerate
partitions), 2 = invalid configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from fraud_detection.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from fraud_detection.data.split import (
    HOLDOUT,
    PARTITION_FILENAMES,
    SplitError,
    SplitResult,
    load_clean_data,
    split_dataset,
    write_json,
    write_partitions,
)
from fraud_detection.data.validate import compute_file_sha256

logger = logging.getLogger("fraud_detection.pipeline.split")

EXIT_OK = 0
EXIT_SPLIT_FAILED = 1
EXIT_CONFIG_ERROR = 2


def stage_outputs(config: Config) -> list[Path]:
    """Every file this stage writes."""
    processed = [config.paths.processed_dir / name for name in PARTITION_FILENAMES.values()]
    return [*processed, config.paths.split_summary, config.paths.temporal_split_summary]


def remove_stale_outputs(config: Config) -> None:
    """Delete outputs of a previous run so later stages cannot consume them after a failure."""
    for path in stage_outputs(config):
        if path.exists():
            path.unlink()
            logger.warning("Removed stale output %s", path)


def run_split_stage(config: Config) -> SplitResult:
    """Split the clean dataset and write all stage outputs.

    Raises:
        SplitError: If the input is missing/invalid or a partition would be degenerate.
            Stale outputs from earlier runs are removed before the error propagates.
    """
    try:
        df = load_clean_data(config.paths.interim)
        result = split_dataset(
            df,
            config.schema,
            config.split,
            config.temporal,
            config.project.seed,
            source_path=config.paths.interim,
            sha256=compute_file_sha256(config.paths.interim),
        )
    except SplitError:
        remove_stale_outputs(config)
        raise

    write_partitions(result.partitions, config.paths.processed_dir)
    write_json(result.split_summary, config.paths.split_summary)
    write_json(result.temporal_summary, config.paths.temporal_split_summary)
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carve out the temporal holdout and create stratified train/val/test splits."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns the process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("Invalid configuration: %s", exc)
        return EXIT_CONFIG_ERROR

    try:
        result = run_split_stage(config)
    except SplitError as exc:
        logger.error("Split FAILED: %s", exc)
        return EXIT_SPLIT_FAILED

    s = result.split_summary["splits"]
    h = result.temporal_summary["temporal_holdout"]
    logger.info(
        "Split complete: train %d / val %d / test %d / %s %d rows (holdout frauds: %d%s)",
        s["train"]["rows"], s["val"]["rows"], s["test"]["rows"], HOLDOUT, h["rows"],
        h["fraud_count"], ", LOW CONFIDENCE" if result.temporal_summary["low_confidence"] else "",
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
