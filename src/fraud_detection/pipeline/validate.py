"""DVC stage entrypoint: ``validate`` (DOC-03 §4.2, stage 1).

Reads ``paths.raw``, writes ``paths.validation_report`` on every run and, only when
validation passes, ``paths.interim`` (validated + deduplicated data).

Usage::

    python -m fraud_detection.pipeline.validate [--config config/config.yaml]

Exit codes: 0 = passed, 1 = validation failed (including unreadable data),
2 = invalid configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from fraud_detection.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from fraud_detection.data.validate import (
    CheckResult,
    DataLoadError,
    Status,
    ValidationResult,
    build_report,
    validate_dataset,
    write_clean_data,
    write_report,
)

logger = logging.getLogger("fraud_detection.pipeline.validate")

EXIT_OK = 0
EXIT_VALIDATION_FAILED = 1
EXIT_CONFIG_ERROR = 2


def run_validation_stage(config: Config) -> ValidationResult:
    """Validate the raw dataset and write the stage outputs.

    On a load failure a FAIL report is still written so the report always reflects
    the latest run. On any failure a stale clean dataset is removed so downstream
    stages cannot consume data that no longer passes validation.
    """
    paths = config.paths
    try:
        result = validate_dataset(paths.raw, config.schema, config.validation)
    except DataLoadError as exc:
        logger.error("%s", exc)
        check = CheckResult("file_load", Status.FAIL, str(exc))
        report = build_report([check], None, None, config.schema, source_path=paths.raw)
        result = ValidationResult(report=report, clean_data=None)

    write_report(result.report, paths.validation_report)

    if result.clean_data is not None:
        write_clean_data(result.clean_data, paths.interim)
    elif paths.interim.exists():
        paths.interim.unlink()
        logger.warning("Removed stale clean dataset at %s", paths.interim)
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and deduplicate the raw credit-card transactions dataset."
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml"
    )
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

    result = run_validation_stage(config)
    report = result.report
    if not result.passed:
        logger.error(
            "Validation FAILED with %d error(s); see %s",
            len(report["errors"]),
            config.paths.validation_report,
        )
        return EXIT_VALIDATION_FAILED

    logger.info(
        "Validation PASSED (%d warning(s)): %s raw rows, %s frauds, %s duplicates removed, "
        "%s clean rows",
        len(report["warnings"]),
        report["total_rows"],
        report["fraud_count"],
        report["duplicate_rows"],
        report["clean_dataset"]["total_rows"],
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
