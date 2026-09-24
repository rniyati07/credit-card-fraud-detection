# Temporal Robustness Assessment

> Secondary, evaluation-only experiment (DOC-02 §21). The frozen `xgboost` (version `1.0.0+b8aee2c`) and its frozen threshold 0.9111537337 were evaluated once on the chronologically latest transactions and compared with the primary random test. Nothing was retrained, retuned or reselected; these results never feed back into the model, threshold or r_min (DOC-02 §21.6). Regenerate with `python -m fraud_detection.pipeline.evaluate_temporal`.

## Setup

| Item | Value |
|---|---|
| Model / version | `xgboost` / `1.0.0+b8aee2c` |
| Frozen threshold | 0.9111537337 (recall>=r_min_max_precision) |
| r_min | 0.8 (confirmed) |
| Temporal holdout | 42,561 rows, 52 frauds, prevalence 0.00122 |
| Primary random test | 36,175 rows, 63 frauds, prevalence 0.00174 |
| Development pool (M3) | 241,165 rows, 421 frauds, prevalence 0.00175, Time 0-151319 s |
| Holdout time range | 151320-172792 s (strictly after the pool) |
| Low-confidence flag (< 20 frauds) | no |

## Comparison (DOC-02 §21.5)

| Metric | Temporal holdout | Random test | Abs Δ | Rel Δ | Note |
|---|---|---|---|---|---|
| pr_auc | 0.7669 | 0.8834 | -0.11647 | -13.18% |  |
| roc_auc | 0.9504 | 0.9779 | -0.02746 | -2.81% |  |
| precision | 0.9512 | 0.9811 | -0.02991 | -3.05% |  |
| recall | 0.7500 | 0.8254 | -0.07540 | -9.13% |  |
| f1 | 0.8387 | 0.8966 | -0.05784 | -6.45% |  |
| f2 | 0.7831 | 0.8525 | -0.06933 | -8.13% |  |
| tp | 39 | 52 | - | - | sizes differ: counts reported side by side |
| fp | 2 | 1 | - | - | sizes differ: counts reported side by side |
| tn | 42,507 | 36,111 | - | - | sizes differ: counts reported side by side |
| fn | 13 | 11 | - | - | sizes differ: counts reported side by side |
| fp_rate | 0.00005 | 0.00003 | 0.00002 | - |  |
| fn_rate | 0.2500 | 0.1746 | 0.07540 | - |  |
| prevalence | 0.00122 | 0.00174 | - | - | context: read PR-AUC against each set's prevalence |
| pr_auc_lift | 627.7272 | 507.2593 | 120.46789 | - | PR-AUC / prevalence (lift over no-skill) |

Bootstrap 95% CIs (1000 seeded resamples per set):

| Metric | Temporal holdout CI | Random test CI | Overlap |
|---|---|---|---|
| pr_auc | [0.6587, 0.8723] | [0.7976, 0.9547] | yes |
| precision | [0.8750, 1.0000] | [0.9355, 1.0000] | yes |
| recall | [0.6308, 0.8605] | [0.7272, 0.9153] | yes |

Confusion matrices at the frozen threshold (rows: actual legitimate/fraud; columns: predicted legitimate/fraud):

- Temporal holdout: [[42507, 2], [13, 39]]
- Random test: [[36111, 1], [11, 52]]

## Answers

**1. Does the model generalise to future transactions?**

On the latest 15% of this 2-day capture: PR-AUC 0.7669 vs 0.8834 on the random test (lift over no-skill 628x vs 507x), recall 0.7500 vs 0.8254, precision 0.9512 vs 0.9811. Overall: lower point estimates on the holdout for pr_auc, recall, precision, but every decrease is within sampling noise (inconclusive at this sample size): no evidence beyond noise of possible short-horizon temporal/distribution shift. Holdout recall is below r_min = 0.8 (margin -3 fraud case(s)). This covers hours, not months, so it indicates short-horizon behaviour only (DOC-01 L-01).

**2. How much did performance change?** (temporal − test)

- pr_auc: -0.1165 (-13.18%) -> decrease within sampling noise (inconclusive) (bootstrap CIs overlap)
- recall: -0.0754 (-9.13%) -> decrease within sampling noise (inconclusive) (bootstrap CIs overlap)
- precision: -0.0299 (-3.05%) -> decrease within sampling noise (inconclusive) (bootstrap CIs overlap)
- Recall change expressed in holdout fraud cases: -3.9.

**3. Is the degradation acceptable according to the project requirements?**

DOC-01 SC-10 mandates no minimum temporal performance; the requirement is that the holdout is evaluated once with the frozen model and threshold and that the comparison, prevalence and interpretation are reported. Those requirements are met by this report. Whether the observed change is acceptable for a deployment is a business decision outside the documented criteria.

**4. Did recall stay near the M6 target?**

- Validation (M6): recall 0.8095; detected 51 of 63; margin over r_min +0 fraud case(s).
- Random test: recall 0.8254; detected 52 of 63 frauds (missed 11); r_min 0.8 needs 51 -> margin +1 fraud case(s); meets r_min.
- Temporal holdout: recall 0.7500; detected 39 of 52 frauds (missed 13); r_min 0.8 needs 42 -> margin -3 fraud case(s); below r_min.

**5. Does the M6 plateau-edge limitation appear in practice?**

M6 recorded that the frozen threshold sits at the upper edge of the validation recall plateau (validation margin over r_min: +0 fraud case(s)), so unseen data could fall below r_min by one or two frauds (DOC-05 §23 DEC-01). It **does** appear: temporal holdout recall 0.7500 is 3 fraud case(s) short of r_min. Reported as-is; the threshold is not retuned (DOC-02 §14).

## Prevalence context (DOC-01 L-03)

- Holdout prevalence 0.00122 vs random test 0.00174 (difference -0.00052).
- Holdout prevalence vs development pool 0.00175: difference -0.00052 (ratio 0.70). The no-skill PR-AUC baseline differs accordingly, so PR-AUC is read together with its lift.

## Interpretation rules and limitations

- A decrease on the temporal holdout is treated as evidence of possible short-horizon temporal/distribution shift within the capture window, not as proof of drift in a live deployment (DOC-02 §21.6, D-13).
- Decreases whose 95% bootstrap CIs overlap, or recall changes within 2 fraud cases, are treated as inconclusive. Overlapping intervals are a conservative criterion: 'inconclusive' means the data cannot distinguish the change from sampling noise, not that no change occurred.
- Few frauds (52 in the holdout, 63 in the test set) make every metric high-variance (DOC-01 L-02).
- `Time` was used only to order transactions; it is not a model feature (DOC-02 §21.2).
- The model, threshold and r_min were frozen before this evaluation and are not changed by it.
