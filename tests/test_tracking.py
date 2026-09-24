"""Lineage tag tests: ``git_dirty`` scope (DOC-03 §6, DOC-05 §23 DEV-10)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from fraud_detection.tracking.mlflow_utils import git_info, lineage_tags

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    for rel in ("src/code.py", "reports/metrics.json", "models/train_metadata.json"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("v1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_tree(repo: Path) -> None:
    info = git_info(cwd=repo, ignore_paths=("reports", "models"))
    assert info["git_dirty"] == "false" and len(info["git_commit"]) == 40


def test_changed_outputs_are_ignored_only_when_excluded(repo: Path) -> None:
    (repo / "reports" / "metrics.json").write_text("v2\n")
    (repo / "models" / "train_metadata.json").write_text("v2\n")
    (repo / "reports" / "new_output.csv").write_text("x\n")

    assert git_info(cwd=repo, ignore_paths=("reports", "models"))["git_dirty"] == "false"
    assert git_info(cwd=repo)["git_dirty"] == "true"


@pytest.mark.parametrize("change", ["modify", "untracked"])
def test_code_changes_still_make_the_tree_dirty(repo: Path, change: str) -> None:
    if change == "modify":
        (repo / "src" / "code.py").write_text("v2\n")
    else:
        (repo / "src" / "new_module.py").write_text("x\n")
    assert git_info(cwd=repo, ignore_paths=("reports", "models"))["git_dirty"] == "true"


def test_lineage_tags_record_the_excluded_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("x: 1\n")
    tags = lineage_tags(
        config_path=config_path,
        raw_path=tmp_path / "missing.csv",
        validation_report_path=tmp_path / "missing.json",
        stage="train",
        run_type="final",
        output_paths=(Path("reports"), Path("models")),
    )
    assert tags["git_dirty_excludes"] == "reports,models,dvc.lock"
    assert tags["data_dvc_md5"] == "unknown" and tags["data_sha256"] == "unknown"
    assert {"git_commit", "git_dirty", "config_hash", "stage", "run_type"} <= set(tags)


def test_line_ending_only_rewrite_is_not_dirty(repo: Path) -> None:
    # dvc repro rewrites data/raw/*.dvc with CRLF on Windows; content is unchanged (DEV-17).
    (repo / ".gitattributes").write_text("* text=auto eol=lf\n")
    (repo / "data.dvc").write_bytes(b"outs:\n- md5: abc\n  path: data.csv\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "dvc file")
    (repo / "data.dvc").write_bytes(b"outs:\r\n- md5: abc\r\n  path: data.csv\r\n")
    assert git_info(cwd=repo, ignore_paths=("reports", "models"))["git_dirty"] == "false"
    (repo / "data.dvc").write_bytes(b"outs:\r\n- md5: CHANGED\r\n  path: data.csv\r\n")
    assert git_info(cwd=repo, ignore_paths=("reports", "models"))["git_dirty"] == "true"


def test_staged_changes_are_dirty(repo: Path) -> None:
    (repo / "src" / "code.py").write_text("v2\n")
    _git(repo, "add", "src/code.py")
    assert git_info(cwd=repo, ignore_paths=("reports", "models"))["git_dirty"] == "true"


def test_dvc_lock_changes_do_not_make_the_tree_dirty(repo: Path) -> None:
    (repo / "dvc.lock").write_text("schema: '2.0'\n")
    _git(repo, "add", "dvc.lock")
    _git(repo, "commit", "-q", "-m", "lock")
    (repo / "dvc.lock").write_text("schema: '2.0'\nstages: {}\n")  # rewritten by dvc repro
    assert git_info(cwd=repo, ignore_paths=("reports", "models", "dvc.lock"))["git_dirty"] == "false"
    assert git_info(cwd=repo, ignore_paths=("reports", "models"))["git_dirty"] == "true"
