"""DVC pipeline tests (DOC-03 §4.2, MAC-01..04; DOC-05 M8, §8).

The pipeline definition is checked against the code, not against a copy of itself: declared
outputs must equal what each stage's modules write, code dependencies must cover everything a
stage imports, and parameters must exist in ``config/config.yaml``. CLI checks run only where
DVC and the pipeline state are available.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from fraud_detection.config import load_config
from fraud_detection.pipeline import evaluate, evaluate_temporal, select, split, tune

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STAGES = ["validate", "split", "train", "evaluate", "evaluate_temporal"]
STAGE_MODULES = {
    "validate": ["fraud_detection.pipeline.validate"],
    "split": ["fraud_detection.pipeline.split"],
    "train": ["fraud_detection.pipeline.tune", "fraud_detection.pipeline.select"],
    "evaluate": ["fraud_detection.pipeline.evaluate"],
    "evaluate_temporal": ["fraud_detection.pipeline.evaluate_temporal"],
}
# MAC-03: metrics must reproduce within floating-point tolerance (DOC-02 §16, DOC-03 §8).
METRIC_TOLERANCE = 1e-9
DVC = shutil.which("dvc") or (str(ROOT / "venv" / "Scripts" / "dvc.exe") if (ROOT / "venv" / "Scripts" / "dvc.exe").exists() else None)


@pytest.fixture(scope="module")
def pipeline() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config" / "config.yaml")


def _entries(stage: dict[str, Any], section: str) -> dict[str, dict[str, Any]]:
    """``{path: options}`` for a stage's outs/metrics/plots (string or single-key mapping entries)."""
    out: dict[str, dict[str, Any]] = {}
    for entry in stage.get(section, []):
        if isinstance(entry, str):
            out[entry] = {}
        else:
            ((path, options),) = entry.items()
            out[path] = options or {}
    return out


def _outputs(stage: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {**_entries(stage, "outs"), **_entries(stage, "metrics"), **_entries(stage, "plots")}


def _deps(stage: dict[str, Any]) -> set[str]:
    return set(stage.get("deps", []))


def _posix(paths) -> set[str]:
    return {Path(p).as_posix() for p in paths}


# ------------------------------------------------------------------------------ structure


def test_exactly_the_five_documented_stages(pipeline) -> None:
    # DOC-03 MD-05; EDA and the M4 baseline are not stages (MD-16, DOC-05 M8 "Must NOT add stages").
    assert list(pipeline["stages"]) == STAGES


def test_stage_commands(pipeline) -> None:
    stages = pipeline["stages"]
    for name in ("validate", "split", "evaluate", "evaluate_temporal"):
        assert stages[name]["cmd"] == f"python -m fraud_detection.pipeline.{name}"
    # DEV-07 / DEV-09: the train stage runs tuning, then threshold optimisation and selection.
    assert stages["train"]["cmd"] == [
        "python -m fraud_detection.pipeline.tune",
        "python -m fraud_detection.pipeline.select",
    ]


def test_structural_leakage_guards(pipeline) -> None:
    stages = pipeline["stages"]
    # MAC-04 / MD-06: train never sees test or holdout; evaluate never sees the holdout.
    assert not {"data/processed/test.csv", "data/processed/temporal_holdout.csv"} & _deps(stages["train"])
    assert "data/processed/temporal_holdout.csv" not in _deps(stages["evaluate"])
    assert {"data/processed/train.csv", "data/processed/val.csv"} <= _deps(stages["train"])
    assert "data/processed/test.csv" in _deps(stages["evaluate"])
    temporal = _deps(stages["evaluate_temporal"])
    assert {"data/processed/temporal_holdout.csv", "reports/test_metrics.json"} <= temporal
    assert "reports/temporal/temporal_split_summary.json" in temporal  # DEV-12
    assert "data/processed/test.csv" not in temporal and "data/processed/val.csv" not in temporal


def test_declared_outputs_equal_what_each_stage_writes(pipeline, config) -> None:
    expected = {
        "validate": {config.paths.interim, config.paths.validation_report},
        "split": set(split.stage_outputs(config)),
        "train": set(tune.stage_outputs(config)) | set(select.stage_outputs(config)),
        "evaluate": set(evaluate.stage_outputs(config)),
        "evaluate_temporal": set(evaluate_temporal.stage_outputs(config)),
    }
    for name, paths in expected.items():
        assert set(_outputs(pipeline["stages"][name])) == _posix(paths), name


def test_no_output_is_declared_twice(pipeline) -> None:
    seen: dict[str, str] = {}
    for name, stage in pipeline["stages"].items():
        for path in _outputs(stage):
            assert path not in seen, f"{path} declared by {seen.get(path)} and {name}"
            seen[path] = name
    assert seen["models/model_config.json"] == "evaluate_temporal"  # DOC-05 §8


def test_dependencies_exist_or_come_from_upstream(pipeline) -> None:
    produced: set[str] = {"data/raw/creditcard.csv"}  # DVC-tracked via creditcard.csv.dvc
    for name in STAGES:
        stage = pipeline["stages"][name]
        for dep in _deps(stage):
            assert dep in produced or (ROOT / dep).exists(), f"{name}: dependency {dep} missing"
            if dep.startswith("data/") or dep.startswith("models/"):
                assert dep in produced, f"{name}: data/model dependency {dep} is not produced upstream"
        produced |= set(_outputs(stage))


def _module_path(module: str) -> Path:
    base = SRC / Path(*module.split("."))
    return base.with_suffix(".py") if base.with_suffix(".py").exists() else base / "__init__.py"


def _transitive_imports(modules: list[str]) -> set[Path]:
    seen: set[Path] = set()
    todo = [_module_path(m) for m in modules]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for module in re.findall(r"^from (fraud_detection[\w.]*) import", path.read_text(encoding="utf-8"), re.M):
            target = _module_path(module)
            if target.name != "__init__.py" or target.parent != SRC / "fraud_detection":
                todo.append(target)
    return seen


def test_code_dependencies_cover_all_imported_modules(pipeline) -> None:
    for name, modules in STAGE_MODULES.items():
        deps = [ROOT / d for d in _deps(pipeline["stages"][name])]
        for path in _transitive_imports(modules):
            if path.name == "__init__.py":
                continue
            covered = any(path == d or d in path.parents for d in deps)
            assert covered, f"{name}: {path.relative_to(ROOT).as_posix()} is imported but not a dependency"


def _lookup(mapping: dict[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        value = value[part]
    return value


def test_params_exist_in_config(pipeline) -> None:
    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    for name, stage in pipeline["stages"].items():
        for entry in stage["params"]:
            ((file, keys),) = entry.items()
            assert file == "config/config.yaml"  # single parameter source (DOC-03 §5)
            for key in keys:
                _lookup(raw, key)  # raises KeyError if missing


def test_train_params_include_documented_sections(pipeline) -> None:
    ((_, keys),) = pipeline["stages"]["train"]["params"][0].items()
    for key in ("schema", "models", "imbalance", "tuning", "threshold", "project.seed"):
        assert key in keys  # DOC-03 §4.2 row 3


# -------------------------------------------------------------------- artifact policy


def test_cache_policy(pipeline) -> None:
    outputs = {p: o for stage in pipeline["stages"].values() for p, o in _outputs(stage).items()}
    git_tracked = {p for p, o in outputs.items() if o.get("cache") is False}
    cached = set(outputs) - git_tracked
    # DOC-05 §8: metrics with cache false for *metrics.json, validation_report.json, split_summary.json.
    assert {"reports/validation/validation_report.json", "reports/split_summary.json",
            "reports/test_metrics.json", "reports/temporal/temporal_metrics.json"} <= git_tracked
    # DOC-03 §3: data, model binaries and figures are DVC-managed, never Git.
    for path in cached:
        assert not path.endswith((".json", ".md")) or path.startswith("models/"), path
    assert {p for p in outputs if p.endswith((".joblib", ".png")) or p.startswith("data/")} <= cached
    # M7 decision 5 / DEC-03: model metadata and threshold tables are DVC-managed.
    assert {"models/train_metadata.json", "models/model_config.json",
            "reports/threshold_analysis_xgboost.csv"} <= cached


def test_documented_metrics_and_plots(pipeline) -> None:
    metrics = {p for s in pipeline["stages"].values() for p in _entries(s, "metrics")}
    plots = {p for s in pipeline["stages"].values() for p in _entries(s, "plots")}
    assert {"reports/validation/validation_report.json", "reports/split_summary.json",
            "reports/test_metrics.json", "reports/temporal/temporal_metrics.json"} <= metrics
    assert {f"reports/threshold_analysis_{m}.csv" for m in ("logistic_regression", "random_forest", "xgboost")} == plots


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_cached_outputs_are_not_tracked_by_git(pipeline) -> None:
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split())
    cached = {p for s in pipeline["stages"].values() for p, o in _outputs(s).items() if o.get("cache") is not False}
    assert not cached & tracked


def test_line_endings_and_dvcignore() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in attributes  # identical dependency bytes in every clone (MAC-03)
    dvcignore = (ROOT / ".dvcignore").read_text(encoding="utf-8")
    for pattern in ("venv/", "__pycache__/", "*.pyc"):
        assert pattern in dvcignore


# --------------------------------------------------------------------- requirements file


def test_requirements_file_is_portable_and_pinned() -> None:
    raw = (ROOT / "requirements.txt").read_bytes()
    assert not raw.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf"))  # no UTF-16 / BOM
    lines = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip() and not line.startswith("#")]
    assert not [line for line in lines if line.startswith("-e")]  # package installed via `pip install -e .`
    assert all(re.match(r"^[A-Za-z0-9_.\-\[\]]+==[^=;\s]+(;\s*.+)?$", line) for line in lines)  # NFR-008
    pywin32 = [line for line in lines if line.lower().startswith("pywin32==")]
    assert all('sys_platform == "win32"' in line for line in pywin32)


def test_requirements_pin_the_runtime_ml_libraries() -> None:
    import sklearn
    import xgboost

    pins = dict(line.split(";")[0].split("==") for line in
                (ROOT / "requirements.txt").read_text(encoding="utf-8").split() if "==" in line)
    assert pins["scikit-learn"] == sklearn.__version__ and pins["xgboost"] == xgboost.__version__  # MAC-09


# ----------------------------------------------------------------- MAC-03 tolerance rule


def metric_changes_beyond_tolerance(diff: dict[str, Any], tolerance: float = METRIC_TOLERANCE) -> list[str]:
    """Numeric metric changes in ``dvc metrics diff --json`` output that exceed ``tolerance``."""
    failures = []
    for path, metrics in diff.items():
        for name, change in metrics.items():
            delta = change.get("diff")
            if isinstance(delta, (int, float)) and abs(delta) > tolerance:
                failures.append(f"{path}:{name} changed by {delta}")
    return failures


def test_tolerance_rule() -> None:
    diff = {"reports/test_metrics.json": {"pr_auc": {"old": 0.88, "new": 0.88 + 1e-12, "diff": 1e-12},
                                          "recall": {"old": 0.8, "new": 0.7, "diff": -0.1}}}
    assert metric_changes_beyond_tolerance(diff) == ["reports/test_metrics.json:recall changed by -0.1"]
    assert metric_changes_beyond_tolerance({}) == []


# -------------------------------------------------------------------- DVC CLI checks

needs_dvc = pytest.mark.skipif(DVC is None, reason="dvc not installed")
needs_state = pytest.mark.skipif(
    not (ROOT / "dvc.lock").exists() or not (ROOT / "models" / "train_metadata.json").exists(),
    reason="pipeline not reproduced in this checkout (no dvc.lock / outputs)",
)


def _dvc(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([DVC, *args], cwd=ROOT, capture_output=True, text=True, timeout=600)


@needs_dvc
def test_dvc_parses_pipeline_and_dag_has_no_forbidden_edges() -> None:
    result = _dvc("dag", "--outs", "--md")
    assert result.returncode == 0, result.stderr
    edges = re.findall(r'"(.+?)"\s*-->\s*"(.+?)"', result.stdout)
    for upstream, downstream in edges:
        if "temporal_holdout.csv" in upstream:
            assert "temporal" in downstream or "model_config" in downstream, (upstream, downstream)


@needs_dvc
@needs_state
def test_dvc_status_and_metrics_commands_work() -> None:
    status = _dvc("status", "--json")
    assert status.returncode == 0, status.stderr
    json.loads(status.stdout or "{}")
    show = _dvc("metrics", "show", "--json")
    assert show.returncode == 0, show.stderr
    shown = json.dumps(json.loads(show.stdout)).replace("\\\\", "/")  # DVC prints OS paths on Windows
    for path in ("reports/test_metrics.json", "reports/temporal/temporal_metrics.json", "models/model_config.json"):
        assert path in shown


@needs_dvc
@needs_state
def test_dvc_metrics_diff_within_tolerance() -> None:
    diff = _dvc("metrics", "diff", "--json")
    assert diff.returncode == 0, diff.stderr
    changes = json.loads(diff.stdout or "{}")
    # Compares HEAD with the workspace. Before dvc.yaml is committed every value is new ("old": null, no
    # "diff"); after a reproduction of the committed pipeline nothing may move beyond tolerance (MAC-03).
    assert metric_changes_beyond_tolerance(changes) == []


@needs_dvc
@needs_state
def test_pipeline_is_up_to_date_and_nothing_reruns() -> None:
    status = _dvc("status", "--json")
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout or "{}") == {}  # MAC-02
    dry = _dvc("repro", "--dry")
    assert dry.returncode == 0, dry.stderr
    assert "Running stage" not in dry.stdout  # no unintended stage execution
