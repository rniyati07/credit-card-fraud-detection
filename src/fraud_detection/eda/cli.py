"""Run the descriptive EDA end to end (DOC-05 M2).

Reads ``paths.interim`` (clean data) and, if present, ``paths.validation_report``;
writes ``<paths.eda_dir>/eda_report.md``, ``eda_summary.json`` and ``figures/*.png``.
EDA is not a DVC stage and nothing downstream reads its outputs (DOC-03 MD-16).

Usage::

    python -m fraud_detection.eda [--config config/config.yaml]

Exit codes: 0 = success, 1 = input data missing/invalid, 2 = invalid configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fraud_detection.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from fraud_detection.eda.analysis import EdaInputError, EdaResults, compute_eda, load_clean_data
from fraud_detection.eda.plots import generate_figures
from fraud_detection.eda.report import render_report

logger = logging.getLogger("fraud_detection.eda")

REPORT_FILENAME = "eda_report.md"
SUMMARY_FILENAME = "eda_summary.json"
FIGURES_DIRNAME = "figures"

EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_CONFIG_ERROR = 2


@dataclass(frozen=True)
class EdaOutputs:
    """Paths of the files written by :func:`run_eda`."""

    report: Path
    summary: Path
    figures: dict[str, Path]


def load_validation_report(path: Path) -> dict[str, Any] | None:
    """Return the M1 validation report, or ``None`` (with a warning) if unavailable."""
    if not path.is_file():
        logger.warning("Validation report not found at %s; M1 reconciliation skipped", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Validation report at %s is not valid JSON (%s); skipped", path, exc)
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_summary(results: EdaResults, figures: dict[str, Path], path: Path) -> Path:
    """Write the machine-readable EDA summary (deterministic key order, no timestamp)."""
    payload = results.to_dict()
    payload["figures"] = {name: p.relative_to(path.parent).as_posix() for name, p in figures.items()}
    _write_text(path, json.dumps(payload, indent=2, default=_json_default, allow_nan=False) + "\n")
    logger.info("EDA summary written to %s", path)
    return path


def run_eda(config: Config) -> EdaOutputs:
    """Compute all analyses, render figures, and write the report and summary.

    Raises:
        EdaInputError: If the clean dataset is missing or lacks required columns.
    """
    df = load_clean_data(config.paths.interim)
    validation_report = load_validation_report(config.paths.validation_report)
    results = compute_eda(df, config.schema, config.eda, validation_report)

    eda_dir = config.paths.eda_dir
    figures = generate_figures(df, results, config.schema, config.eda, eda_dir / FIGURES_DIRNAME)

    report_path = eda_dir / REPORT_FILENAME
    markdown = render_report(
        results, figures, eda_dir, config.schema, config.paths.interim, config.eda.top_k_features
    )
    _write_text(report_path, markdown)
    logger.info("EDA report written to %s", report_path)

    summary_path = write_summary(results, figures, eda_dir / SUMMARY_FILENAME)
    return EdaOutputs(report=report_path, summary=summary_path, figures=figures)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descriptive EDA of the deduplicated dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns the process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )
    # Third-party libraries are chatty at DEBUG (font discovery, PNG chunks).
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLogger().level))

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("Invalid configuration: %s", exc)
        return EXIT_CONFIG_ERROR

    try:
        outputs = run_eda(config)
    except EdaInputError as exc:
        logger.error("%s", exc)
        return EXIT_INPUT_ERROR

    logger.info("EDA complete: %s, %d figures", outputs.report, len(outputs.figures))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
