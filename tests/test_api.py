"""API tests with FastAPI TestClient and the fixture artifact pair (DOC-04 §12; DOC-05 §13, §15).

Requires ``httpx`` (FastAPI's TestClient transport).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app, check_schema_compatibility
from app.schemas import TransactionRequest
from conftest import FEATURES, write_serving_artifacts
from fraud_detection.inference.predictor import FraudPredictor, ModelLoadError

ROOT = Path(__file__).resolve().parents[1]
RESPONSE_FIELDS = {"fraud_probability", "prediction", "label", "threshold", "model_name", "model_version"}


@pytest.fixture
def client(serving_artifacts, monkeypatch):
    model_path, config_path, _ = serving_artifacts
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_CONFIG_PATH", str(config_path))
    with TestClient(app) as test_client:
        yield test_client


def _post_raw(client: TestClient, body: str):
    return client.post("/predict", content=body, headers={"Content-Type": "application/json"})


def _error_fields(response) -> set[str]:
    return {str(part) for error in response.json()["detail"] for part in error["loc"]}


# ------------------------------------------------------------------------------ happy path


def test_health(client, serving_artifacts) -> None:
    _, _, config = serving_artifacts
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "model_loaded": True,
        "model_name": config["model_name"], "model_version": config["model_version"],
    }


def test_predict_valid(client, transaction, serving_artifacts) -> None:
    _, _, config = serving_artifacts
    response = client.post("/predict", json=transaction)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_FIELDS  # exactly the documented fields
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["threshold"] == config["threshold"]
    assert body["prediction"] == int(body["fraud_probability"] >= body["threshold"])
    assert body["label"] == ("fraud" if body["prediction"] == 1 else "legitimate")
    assert (body["model_name"], body["model_version"]) == (config["model_name"], config["model_version"])
    assert isinstance(body["prediction"], int) and isinstance(body["fraud_probability"], float)


def test_integers_are_valid_json_numbers(client, transaction) -> None:
    payload = {k: round(v) for k, v in transaction.items()} | {"Amount": 10}
    assert client.post("/predict", json=payload).status_code == 200


def test_zero_amount_accepted(client, transaction) -> None:
    assert client.post("/predict", json=transaction | {"Amount": 0.0}).status_code == 200


def test_key_order_independence(client, transaction) -> None:
    forward = client.post("/predict", json=transaction).json()
    backward = client.post("/predict", json=dict(reversed(list(transaction.items())))).json()
    assert forward == backward


def test_openapi_docs_render(client) -> None:
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    request_schema = schema["components"]["schemas"]["TransactionRequest"]
    assert set(request_schema["properties"]) == set(FEATURES)
    assert request_schema.get("additionalProperties") is False


# --------------------------------------------------------------------------- validation (422)


def test_missing_feature_rejected(client, transaction) -> None:
    payload = dict(transaction)
    del payload["V5"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 and "V5" in _error_fields(response)


@pytest.mark.parametrize("extra", ["Time", "Class", "foo"])
def test_extra_field_rejected(client, transaction, extra) -> None:
    response = client.post("/predict", json=transaction | {extra: 123})
    assert response.status_code == 422 and extra in _error_fields(response)


@pytest.mark.parametrize("value", ["1.2", "abc", True, None, [1.0], {"v": 1.0}])
def test_non_numeric_values_rejected(client, transaction, value) -> None:
    response = client.post("/predict", json=transaction | {"V1": value})
    assert response.status_code == 422 and "V1" in _error_fields(response)


def test_string_amount_rejected(client, transaction) -> None:
    assert client.post("/predict", json=transaction | {"Amount": "149.62"}).status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_rejected(client, transaction, token) -> None:
    body = json.dumps(transaction | {"V2": 0.0}).replace('"V2": 0.0', f'"V2": {token}')
    response = _post_raw(client, body)
    assert response.status_code == 422 and "V2" in _error_fields(response)


def test_validation_errors_name_fields_but_do_not_echo_values(client, transaction) -> None:
    # DOC-04 §5.3: standard {"detail": [...]} shape, field names only; submitted values never echoed.
    response = client.post("/predict", json=transaction | {"V3": "secret-value-123", "Time": 424242})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert all(set(error) == {"loc", "msg", "type"} for error in errors)
    assert {"V3", "Time"} <= _error_fields(response)
    assert "secret-value-123" not in response.text and "424242" not in response.text


def test_negative_amount_rejected(client, transaction) -> None:
    response = client.post("/predict", json=transaction | {"Amount": -0.01})
    assert response.status_code == 422 and "Amount" in _error_fields(response)


def test_only_time_rejected(client) -> None:
    assert client.post("/predict", json={"Time": 123}).status_code == 422


def test_malformed_json_rejected(client) -> None:
    assert _post_raw(client, '{"V1": 1.0,').status_code == 422


def test_empty_and_non_object_bodies_rejected(client) -> None:
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json=[1, 2, 3]).status_code == 422


def test_wrong_method_and_unknown_path(client) -> None:
    assert client.get("/predict").status_code == 405
    assert client.get("/unknown").status_code == 404


def test_schema_has_exactly_the_model_features() -> None:
    assert list(TransactionRequest.model_fields) == FEATURES
    assert "Time" not in TransactionRequest.model_fields


# ------------------------------------------------------------------------ errors and logging


def test_prediction_failure_returns_generic_500(client, transaction, monkeypatch) -> None:
    def boom(features):
        raise RuntimeError("internal detail /secret/path xgboost 3.4.1")

    monkeypatch.setattr(client.app.state.predictor, "predict", boom)
    response = client.post("/predict", json=transaction)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
    assert "secret" not in response.text and "Traceback" not in response.text


def test_payload_and_prediction_are_not_logged(client, transaction, caplog) -> None:
    sentinel = 987654.321123
    caplog.set_level(logging.DEBUG)
    body = client.post("/predict", json=transaction | {"Amount": sentinel}).json()
    logged = caplog.text
    assert str(sentinel) not in logged
    assert repr(body["fraud_probability"]) not in logged and str(body["fraud_probability"]) not in logged
    assert all(str(round(v, 6)) not in logged for v in transaction.values())


# ------------------------------------------------------------------------ fail-fast startup


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("remove_model", "Model artifact not found"),
        ("time_in_features", "Feature schema incompatible"),
        ("bad_threshold", "Invalid threshold"),
    ],
)
def test_startup_fails_fast(tmp_path, monkeypatch, change, message) -> None:
    overrides = {
        "time_in_features": {"feature_order": ["Time", *FEATURES[1:]]},
        "bad_threshold": {"threshold": 1.0},
    }.get(change, {})
    model_path, config_path, _ = write_serving_artifacts(tmp_path / "bad", **overrides)
    if change == "remove_model":
        model_path.unlink()
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_CONFIG_PATH", str(config_path))
    with pytest.raises(ModelLoadError, match=message):
        with TestClient(app):
            pass


def test_schema_compatibility_check() -> None:
    class Stub:
        feature_order = (*FEATURES[:-1], "V29")

    with pytest.raises(ModelLoadError, match="does not match the API request schema"):
        check_schema_compatibility(Stub())  # type: ignore[arg-type]


def test_default_artifact_paths(monkeypatch) -> None:
    from app.main import artifact_paths

    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_CONFIG_PATH", raising=False)
    assert artifact_paths() == (Path("models/model.joblib"), Path("models/model_config.json"))


# ------------------------------------------------------------- real artifacts (optional)

REAL_MODEL = ROOT / "models" / "model.joblib"
REAL_CONFIG = ROOT / "models" / "model_config.json"
REAL_TEST = ROOT / "data" / "processed" / "test.csv"


@pytest.mark.artifact
@pytest.mark.skipif(not (REAL_MODEL.exists() and REAL_CONFIG.exists() and REAL_TEST.exists()),
                    reason="real DVC outputs not present (run `dvc checkout` or `dvc repro`)")
def test_real_artifact_api_parity(monkeypatch) -> None:
    import pandas as pd

    monkeypatch.setenv("MODEL_PATH", str(REAL_MODEL))
    monkeypatch.setenv("MODEL_CONFIG_PATH", str(REAL_CONFIG))
    config = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    predictor = FraudPredictor.load(REAL_MODEL, REAL_CONFIG)
    row = pd.read_csv(REAL_TEST).iloc[0]
    payload = {f: float(row[f]) for f in config["feature_order"]}
    with TestClient(app) as real_client:
        body = real_client.post("/predict", json=payload).json()  # DR-04
        assert real_client.post("/predict", json=payload | {"Time": float(row["Time"])}).status_code == 422
    assert body["threshold"] == config["threshold"] == 0.9111537337303162
    assert body["model_version"] == config["model_version"]
    assert body["fraud_probability"] == pytest.approx(
        float(predictor.predict_proba(predictor.build_frame(payload))[0]), abs=1e-12
    )
