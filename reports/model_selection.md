# Model Selection and Validation Threshold

> Validation data only. The test split and the temporal holdout were not read. Selection is rule-based (DOC-02 §12); the threshold is chosen per DOC-02 §13. Regenerate with `python -m fraud_detection.pipeline.select`.

## Configuration

| Setting | Value |
|---|---|
| `r_min` | 0.8 (confirmed; fixed for the rest of the project, DOC-05 §23) |
| Threshold grid | 0.01–0.99, step 0.01 |
| PR-curve thresholds included | yes |
| Fallback objective | f2 |
| PR-AUC tie tolerance | 0.01 (provisional) |
| Reference threshold | 0.5 (reported only) |
| Validation rows / frauds | 36,175 / 63 (prevalence 0.00174) |

## Candidates (validation)

| Model | PR-AUC | ROC-AUC | Lift | Reaches r_min | Threshold | Objective | Precision | Recall | F1 | F2 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost **(selected)** | 0.8276 | 0.9665 | 475x | yes | 0.911154 | recall>=r_min_max_precision | 0.9808 | 0.8095 | 0.8870 | 0.8388 | 51 | 1 | 12 |
| random_forest | 0.8145 | 0.9466 | 468x | yes | 0.302499 | recall>=r_min_max_precision | 0.9444 | 0.8095 | 0.8718 | 0.8333 | 51 | 3 | 12 |
| logistic_regression | 0.6662 | 0.9731 | 383x | yes | 0.949264 | recall>=r_min_max_precision | 0.4636 | 0.8095 | 0.5896 | 0.7044 | 51 | 59 | 12 |

Reference (threshold 0.50, not used for selection):

| Model | Precision | Recall | F1 | F2 | FP | FN |
|---|---|---|---|---|---|---|
| xgboost | 0.8947 | 0.8095 | 0.8500 | 0.8252 | 6 | 12 |
| random_forest | 0.9565 | 0.6984 | 0.8073 | 0.7383 | 2 | 19 |
| logistic_regression | 0.0697 | 0.8730 | 0.1291 | 0.2642 | 734 | 8 |

## Selection framework (DOC-02 §12)

- **1. Eligibility**: Eligible (reach recall >= r_min): ['logistic_regression', 'random_forest', 'xgboost']
- **2. Primary ranking (validation PR-AUC)**: xgboost (0.8276), random_forest (0.8145), logistic_regression (0.6662)
- **3. Operating-point check**: Best operating point: xgboost; PR-AUC leader: xgboost
- **4. Tie-break**: No other model within 0.01 PR-AUC of xgboost; no tie-break needed
- **5. LR-baseline sanity**: xgboost PR-AUC 0.8276 > LR baseline 0.6662

## Decision

- **Selected model:** `xgboost` (version `1.0.0+707b4e0`), frozen as `models/model.joblib`.
- **Threshold:** 0.911154 (recall>=r_min_max_precision).
- **At this threshold:** detects 51 of 63 validation frauds (recall 81.0%), misses 12, flags 1 legitimate transaction (precision 98.1%).
- **Granularity:** one validation fraud = 1.6 pp of recall; small differences are noisy.
- **DOC-01 SC-06:** selected PR-AUC 0.8276 vs LR 0.6662 and no-skill 0.00174: pass.

Neighbouring grid thresholds of the selected model (stability check, DOC-02 §13):

| Threshold | Precision | Recall | FP | FN |
|---|---|---|---|---|
| 0.910000 | 0.9808 | 0.8095 | 1 | 12 |
| 0.911154 | 0.9808 | 0.8095 | 1 | 12 |
| 0.920000 | 0.9804 | 0.7937 | 1 | 13 |

## r_min status

`r_min = 0.8` is confirmed (DOC-02 §13 / TD-12 resolved, reason logged in DOC-05 §23) and fixed for the rest of the project. The model, threshold and `r_min` must not change after the test evaluation in M7.
