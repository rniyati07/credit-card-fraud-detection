# Serving image for the frozen fraud model (DOC-04 §8; DOC-05 M10, §14).
#
# Build precondition: models/model.joblib and models/model_config.json must exist
# (`dvc repro` or `dvc pull`). The build never runs DVC, training or MLflow, and never
# copies data/, mlruns/, mlflow.db, reports/, notebooks/, tests/ or training code.
#
#   V=$(python -c "import json; print(json.load(open('models/model_config.json'))['model_version'])")
#   docker build --build-arg MODEL_VERSION="$V" -t "fraud-detector:${V//+/_}" .
#   docker run --rm -p 8000:8000 "fraud-detector:${V//+/_}"

# Python X.Y equals the training Python recorded in model_config.json (python_version 3.14.7).
FROM python:3.14.7-slim

# Exact model_version from model_config.json (DR-08). Docker tags cannot contain "+", so the tag
# uses "_" instead (1.0.0+abc1234 -> fraud-detector:1.0.0_abc1234); the label keeps the exact value.
ARG MODEL_VERSION=unknown
LABEL org.opencontainers.image.title="fraud-detector" \
      org.opencontainers.image.version="${MODEL_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_CONFIG_PATH=/app/models/model_config.json

WORKDIR /app

# Dependencies first (layer cache). --no-deps: requirements-serve.txt pins the complete runtime
# closure, so pip installs exactly those versions and nothing else (DOC-05 §23 DEV-23).
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --no-deps -r requirements-serve.txt

# Serving code only: the FastAPI app and the inference package (DOC-03 §11 import rule).
COPY app/ app/
COPY src/fraud_detection/__init__.py src/fraud_detection/__init__.py
COPY src/fraud_detection/inference/ src/fraud_detection/inference/

# Frozen artifacts only (read-only for the service; loaded once at startup).
COPY models/model.joblib models/model_config.json models/

# Fail the build early if the pinned closure is incomplete (imports only; the model is loaded
# and validated at container startup by the DOC-04 §4 checks).
RUN python -c "import app.main, sklearn, xgboost, pandas, numpy, joblib, uvicorn"

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8000

# Liveness via the Python standard library (no curl in slim images).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# Single worker (DOC-04 §8).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
