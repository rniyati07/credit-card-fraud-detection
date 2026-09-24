"""Fraud-scoring API (DOC-04): ``GET /health`` and ``POST /predict`` (+ built-in ``/docs``).

* Artifacts are loaded once at startup (lifespan) from ``MODEL_PATH`` / ``MODEL_CONFIG_PATH``
  (defaults ``models/model.joblib`` / ``models/model_config.json``). Any failed DOC-04 §4 check
  aborts startup: the service never runs half-loaded.
* No retraining, refitting, threshold recomputation or artifact writes, ever.
* Request payloads, feature values and predictions are never logged; errors never expose
  stack traces or internals to clients (DOC-04 §7).

Run: ``uvicorn app.main:app --host 0.0.0.0 --port 8000``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import HealthResponse, PredictionResponse, TransactionRequest
from fraud_detection.inference.predictor import FraudPredictor, ModelLoadError

logger = logging.getLogger("uvicorn.error")  # shares uvicorn's handler/format; operational messages only

DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_MODEL_CONFIG_PATH = "models/model_config.json"


def artifact_paths() -> tuple[Path, Path]:
    """Artifact locations from ``MODEL_PATH`` / ``MODEL_CONFIG_PATH`` (DOC-04 §4, DD-10)."""
    return (
        Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)),
        Path(os.environ.get("MODEL_CONFIG_PATH", DEFAULT_MODEL_CONFIG_PATH)),
    )


def check_schema_compatibility(predictor: FraudPredictor) -> None:
    """DOC-04 §4 check 3 (API part): ``feature_order`` must equal the request schema's field set.

    Raises:
        ModelLoadError: If they differ.
    """
    fields = set(TransactionRequest.model_fields)
    if set(predictor.feature_order) != fields:
        raise ModelLoadError(
            "Feature schema incompatible: feature_order does not match the API request schema "
            f"(missing in schema: {sorted(set(predictor.feature_order) - fields)}, "
            f"missing in feature_order: {sorted(fields - set(predictor.feature_order))})"
        )


def load_predictor() -> FraudPredictor:
    """Load and fully validate the frozen artifacts (fail fast)."""
    model_path, config_path = artifact_paths()
    predictor = FraudPredictor.load(model_path, config_path)
    check_schema_compatibility(predictor)
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        app.state.predictor = load_predictor()
    except ModelLoadError as exc:
        logger.error("Startup failed: %s", exc)
        raise
    logger.info(
        "Model loaded: %s (version %s)", app.state.predictor.model_name, app.state.predictor.model_version
    )
    yield
    app.state.predictor = None


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Scores one transaction with the frozen fraud model and its validation-selected threshold.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 in FastAPI's standard ``{"detail": [...]}`` shape, naming fields but never echoing the
    submitted values (DOC-04 §5.3). Echoing them would also make NaN/Infinity inputs unserialisable."""
    errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Generic 500: no stack trace, paths, versions or model internals in the response (DOC-04 §7)."""
    logger.error("Unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor: FraudPredictor = request.app.state.predictor
    return HealthResponse(
        status="ok", model_loaded=True, model_name=predictor.model_name, model_version=predictor.model_version
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest, request: Request) -> PredictionResponse | JSONResponse:
    predictor: FraudPredictor = request.app.state.predictor
    try:
        result = predictor.predict(transaction.model_dump())
    except Exception as exc:
        # Traceback is kept server-side; the exception type only, never the payload or its values.
        logger.error("Prediction failed (%s)", type(exc).__name__, exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Prediction failed"})
    return PredictionResponse(**asdict(result))
