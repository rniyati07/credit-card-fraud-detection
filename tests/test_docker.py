"""Static checks of the serving image definition (DOC-04 §8; DOC-05 M10, §14; DOC-03 MAC-09).

The image itself is verified by building and running it (DOC-04 DR-05/06/08); these tests keep
the Dockerfile, .dockerignore and requirements-serve.txt consistent with the rules.
"""

from __future__ import annotations

import json
import re
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
DIRECT_RUNTIME = {"fastapi", "uvicorn", "pydantic", "scikit-learn", "xgboost", "numpy", "pandas", "joblib"}
FORBIDDEN_IN_SERVE = {"mlflow", "dvc", "matplotlib", "seaborn", "pytest", "httpx", "httpcore",
                      "imbalanced-learn", "pywin32", "pyyaml"}
LINUX = {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix", "platform_machine": "x86_64",
         "python_version": "3.14", "implementation_name": "cpython", "platform_python_implementation": "CPython",
         "extra": ""}
ALLOWED_UNPINNED = {"nvidia-nccl-cu13"}  # xgboost's Linux multi-GPU library; not used for CPU inference (DEV-23)
ALLOWED_COPY_SOURCES = {
    "requirements-serve.txt", "app/", "src/fraud_detection/__init__.py", "src/fraud_detection/inference/",
    "models/model.joblib", "models/model_config.json",
}


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split(";")[0].split("==")
            pins[canonicalize_name(name)] = version.strip()
    return pins


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text(encoding="utf-8")


# --------------------------------------------------------------------- requirements-serve


def test_serve_pins_are_identical_to_requirements() -> None:
    serve, full = _pins(ROOT / "requirements-serve.txt"), _pins(ROOT / "requirements.txt")
    assert {canonicalize_name(n) for n in DIRECT_RUNTIME} <= set(serve)
    for name, version in serve.items():
        assert full.get(name) == version, f"{name}: serve {version} vs requirements.txt {full.get(name)}"  # MAC-09


def test_serve_has_no_training_or_tooling_packages() -> None:
    serve = set(_pins(ROOT / "requirements-serve.txt"))
    assert not serve & {canonicalize_name(n) for n in FORBIDDEN_IN_SERVE}


def test_serve_pins_the_complete_linux_runtime_closure() -> None:
    # The image installs with --no-deps, so every runtime dependency must be pinned explicitly.
    serve = _pins(ROOT / "requirements-serve.txt")
    for name in serve:
        try:
            requires = metadata.distribution(name).requires or []
        except metadata.PackageNotFoundError:
            pytest.skip(f"{name} not installed in this environment")
        for raw in requires:
            req = Requirement(raw)
            if req.marker is None or req.marker.evaluate(LINUX):
                dep = canonicalize_name(req.name)
                assert dep in serve or dep in ALLOWED_UNPINNED, f"{name} needs {dep}, not pinned"


# ----------------------------------------------------------------------------- Dockerfile


def test_base_image_matches_training_python(dockerfile) -> None:
    match = re.search(r"^FROM python:(\d+\.\d+)(\.\d+)?-slim\s*$", dockerfile, re.M)
    assert match, "base image must be python:<X.Y[.Z]>-slim"
    config = ROOT / "models" / "model_config.json"
    if config.exists():  # DOC-04 §8: X.Y equals the training Python in model_config.json
        trained = json.loads(config.read_text(encoding="utf-8")).get("python_version", "")
        assert trained.startswith(match.group(1) + "."), (trained, match.group(0))


def test_copies_only_serving_code_and_frozen_artifacts(dockerfile) -> None:
    sources = []
    for line in re.findall(r"^COPY\s+(.+)$", dockerfile, re.M):
        parts = [p for p in line.split() if not p.startswith("--")]
        sources += parts[:-1]
    assert set(sources) == ALLOWED_COPY_SOURCES
    assert "COPY . " not in dockerfile and "ADD " not in dockerfile


def test_runtime_settings(dockerfile) -> None:
    assert re.search(r"^USER\s+(?!root\b|0\b)\S+", dockerfile, re.M)  # non-root
    assert re.search(r"^EXPOSE 8000\s*$", dockerfile, re.M)
    assert 'CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]' in dockerfile
    assert "--workers" not in dockerfile  # single worker (DOC-04 §8)
    for env in ("PYTHONPATH=/app/src", "MODEL_PATH=/app/models/model.joblib",
                "MODEL_CONFIG_PATH=/app/models/model_config.json"):
        assert env in dockerfile
    assert "pip install --no-cache-dir --no-deps -r requirements-serve.txt" in dockerfile
    assert not re.search(r"\b(dvc|mlflow|fraud_detection\.pipeline)\b", " ".join(re.findall(r"^RUN .+$", dockerfile, re.M)))
    assert "HEALTHCHECK" in dockerfile and "/health" in dockerfile
    # DR-08: exact model_version recorded as an OCI label ("+" is not allowed in Docker tags).
    assert "ARG MODEL_VERSION" in dockerfile
    assert 'org.opencontainers.image.version="${MODEL_VERSION}"' in dockerfile


# ---------------------------------------------------------------------------- .dockerignore


def test_dockerignore_is_an_allow_list() -> None:
    lines = [l.strip() for l in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    assert lines[0] == "*"  # exclude everything by default
    allowed = {l[1:] for l in lines if l.startswith("!")}
    assert allowed == ALLOWED_COPY_SOURCES
    assert {"**/__pycache__", "**/*.py[cod]"} <= set(lines)
