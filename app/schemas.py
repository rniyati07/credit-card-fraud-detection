"""Request/response models for the fraud-scoring API (DOC-04 §5).

* Exactly the 29 model features, all required; anything else (``Time``, ``Class``, unknown
  keys) is rejected (``extra="forbid"``, DD-03).
* Strict numbers: JSON numbers only. Strings (``"1.2"``), booleans and ``null`` are rejected,
  never coerced. ``NaN``/``Infinity`` are rejected (``allow_inf_nan=False``, DD-04).
* ``Amount >= 0`` mirrors the DOC-02 §3 validation rule; ``V1``-``V28`` are anonymised PCA
  components with no justified bounds, so no range checks (DOC-04 §5.3).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

VFeature = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, description="Anonymised PCA component (no range constraints)."),
]
AmountFeature = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0, description="Transaction amount; must be >= 0."),
]

# Illustrative values only (shape of one transaction); not a real prediction input of record.
EXAMPLE_REQUEST = {
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24,
    "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
    "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25, "V21": -0.02,
    "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62,
}


class TransactionRequest(BaseModel):
    """One transaction: exactly ``V1``-``V28`` and ``Amount``. ``Time`` is not accepted."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EXAMPLE_REQUEST], "description": "Illustrative example values."},
    )

    V1: VFeature
    V2: VFeature
    V3: VFeature
    V4: VFeature
    V5: VFeature
    V6: VFeature
    V7: VFeature
    V8: VFeature
    V9: VFeature
    V10: VFeature
    V11: VFeature
    V12: VFeature
    V13: VFeature
    V14: VFeature
    V15: VFeature
    V16: VFeature
    V17: VFeature
    V18: VFeature
    V19: VFeature
    V20: VFeature
    V21: VFeature
    V22: VFeature
    V23: VFeature
    V24: VFeature
    V25: VFeature
    V26: VFeature
    V27: VFeature
    V28: VFeature
    Amount: AmountFeature


class PredictionResponse(BaseModel):
    """DOC-04 §5.2 response."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    fraud_probability: float = Field(ge=0, le=1, description="predict_proba[:, 1] of the frozen model.")
    prediction: Literal[0, 1] = Field(description="1 iff fraud_probability >= threshold.")
    label: Literal["fraud", "legitimate"] = Field(description="Human-readable form of prediction.")
    threshold: float = Field(description="Frozen, validation-selected threshold from model_config.json.")
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """DOC-04 §5.1 response. A responding service always has the model loaded (fail-fast startup)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    status: Literal["ok"]
    model_loaded: bool
    model_name: str
    model_version: str
