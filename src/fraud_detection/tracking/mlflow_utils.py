"""MLflow tracking helpers (DOC-03 §6, DOC-05 §9).

Local store only (no server, no Model Registry). Every run carries the DOC-03 §6
lineage tags: ``git_commit``, ``git_dirty``, ``data_sha256``, ``data_dvc_md5``,
``config_hash``, ``stage``, ``run_type``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # suppress an import-time console hint

import mlflow  # noqa: E402
import yaml  # noqa: E402

from fraud_detection.config import MlflowConfig

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"


def git_info(cwd: str | Path | None = None) -> dict[str, str]:
    """Current commit SHA and whether the working tree is dirty ("unknown" without git)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": UNKNOWN, "git_dirty": UNKNOWN}
    return {"git_commit": commit, "git_dirty": str(bool(status.strip())).lower()}


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dvc_md5(raw_path: str | Path) -> str:
    """MD5 recorded by DVC for ``raw_path`` (from ``<raw_path>.dvc``), or "unknown"."""
    dvc_file = Path(f"{raw_path}.dvc")
    try:
        meta = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
        return str(meta["outs"][0]["md5"])
    except (OSError, yaml.YAMLError, KeyError, IndexError, TypeError):
        return UNKNOWN


def raw_data_sha256(validation_report_path: str | Path) -> str:
    """SHA-256 of the raw CSV as recorded by the M1 validation report, or "unknown"."""
    try:
        report = json.loads(Path(validation_report_path).read_text(encoding="utf-8"))
        return str(report["dataset"]["sha256"] or UNKNOWN)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return UNKNOWN


def lineage_tags(
    *,
    config_path: str | Path,
    raw_path: str | Path,
    validation_report_path: str | Path,
    stage: str,
    run_type: str,
) -> dict[str, str]:
    """The DOC-03 §6 run tags."""
    return {
        **git_info(),
        "data_sha256": raw_data_sha256(validation_report_path),
        "data_dvc_md5": dvc_md5(raw_path),
        "config_hash": file_sha256(config_path),
        "stage": stage,
        "run_type": run_type,
    }


def setup_experiment(config: MlflowConfig) -> str:
    """Point MLflow at the configured local store and experiment; return the experiment id."""
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.set_experiment(config.experiment_name)
    logger.info("MLflow experiment '%s' at %s", config.experiment_name, config.tracking_uri)
    return experiment.experiment_id


def _log_params(params: dict[str, Any]) -> None:
    mlflow.log_params({k: "None" if v is None else v for k, v in params.items()})


def log_run(
    *,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str],
    json_artifacts: dict[str, Any] | None = None,
    file_artifacts: list[Path] | None = None,
    children: list[dict[str, Any]] | None = None,
) -> str:
    """Create one run with params, metrics, tags and artifacts; return its run id.

    ``children`` are logged as nested runs of this run; each is a dict with
    ``run_name``, ``params``, ``metrics`` and ``tags``.
    """
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        _log_params(params)
        mlflow.log_metrics(metrics)
        for name, payload in (json_artifacts or {}).items():
            mlflow.log_dict(payload, name)
        for path in file_artifacts or []:
            mlflow.log_artifact(str(path))
        for child in children or []:
            with mlflow.start_run(run_name=child["run_name"], tags=child["tags"], nested=True):
                _log_params(child["params"])
                mlflow.log_metrics(child["metrics"])
    logger.info(
        "MLflow run '%s' logged (id %s, %d nested runs)", run_name, run.info.run_id, len(children or [])
    )
    return run.info.run_id
