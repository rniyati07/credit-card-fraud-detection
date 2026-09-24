# Credit Card Fraud Detection — Project Requirements & Scope

| Field | Value |
|---|---|
| Document ID | DOC-01 |
| Status | Baseline v1.1 (Source of Truth) — adds Temporal Robustness Analysis |
| Companion document | `docs/02_Data_and_ML_Design.md` (DOC-02) |
| Implementation window | ~2 days |
| Change policy | Any deviation from this document must be recorded in §15 with a justification before implementation. |

**Notation used throughout both documents**

| Tag | Meaning |
|---|---|
| **[REQ]** | Mandatory requirement |
| **[DEC]** | Confirmed design decision |
| **[ASM]** | Assumption — must be verified against the actual data/environment |
| **[REC]** | Recommendation — default unless evidence suggests otherwise |
| **[EMP]** | Requires empirical validation during implementation |

---

## 1. Project Overview

**Purpose.** Deliver a compact, end-to-end, reproducible machine-learning system that classifies credit-card transactions as *fraudulent* or *legitimate*, built so that it can be versioned, tracked, served, and containerized in later phases.

**Fraud detection problem.** Fraudulent transactions are rare relative to legitimate ones. A detector must catch as much fraud as possible without flagging so many legitimate transactions that the system becomes operationally unusable.

**ML objective.** Train and compare probabilistic binary classifiers, select a final model using imbalance-appropriate metrics, and choose a decision threshold on validation data rather than defaulting to 0.50.

**MLOps/deployment context.** The ML phase must expose clean boundaries for later integration of Git/GitHub, DVC, MLflow, FastAPI, Docker, Pytest, and (optionally) GitHub Actions.

**Intended outcome.** A single inference-ready model artifact (preprocessing + model), a threshold configuration file, documented evaluation results on an untouched test set, a secondary temporal robustness report on chronologically later transactions, and a reproducible training workflow.

**Experiment structure.** The **primary experiment** (stratified random split) drives model comparison, threshold optimization, and final primary evaluation. A **secondary Temporal Robustness Experiment** evaluates the already-frozen primary model and threshold on a chronologically later holdout. The secondary experiment never influences any training, tuning, or selection decision.

## 2. Problem Statement

Given a transaction feature vector **x**, estimate **p = P(Class = 1 | x)** and output a class decision **ŷ = 1 if p ≥ τ else 0**, where τ is a validation-optimized threshold.

This is a **binary classification / rare-event detection** problem. It differs from balanced classification because:

| Aspect | Balanced classification | Fraud detection |
|---|---|---|
| Class prior | Roughly equal | Positive class is a very small minority |
| Accuracy | Informative | Misleading — predicting "all legitimate" scores near-perfect accuracy |
| Error costs | Often symmetric | Asymmetric: missed fraud (FN) vs. customer friction/review load (FP) |
| Default threshold 0.50 | Often reasonable | Rarely optimal; probabilities are skewed toward class 0 |
| Primary metrics | Accuracy, ROC-AUC | Recall, Precision, F1/F-beta, PR-AUC |

## 3. Objectives

### 3.1 ML objectives

| ID | Objective |
|---|---|
| OBJ-ML-1 | Identify fraudulent transactions with high recall while maintaining usable precision. |
| OBJ-ML-2 | Handle severe class imbalance with a documented, leakage-safe strategy. |
| OBJ-ML-3 | Compare Logistic Regression, Random Forest, and XGBoost under a common protocol. |
| OBJ-ML-4 | Produce calibrated-enough probability outputs, not only hard labels. |
| OBJ-ML-5 | Optimize the decision threshold on validation data against a documented objective. |
| OBJ-ML-6 | Assess temporal robustness: measure whether the frozen primary model and threshold remain effective on chronologically later, fully unseen transactions. |

### 3.2 Engineering/MLOps objectives

| ID | Objective |
|---|---|
| OBJ-ENG-1 | Reproducibility: fixed seeds, configuration-driven runs, pinned dependencies. |
| OBJ-ENG-2 | Versioned data (DVC-ready inputs/outputs). |
| OBJ-ENG-3 | Tracked experiments (MLflow-ready parameters, metrics, artifacts). |
| OBJ-ENG-4 | A reusable, self-contained model artifact. |
| OBJ-ENG-5 | A deployment-ready inference path with identical train/serve preprocessing. |

## 4. Scope

In scope:

| # | Item |
|---|---|
| S-1 | Data inspection of the supplied CSV |
| S-2 | Data cleaning and validation |
| S-3 | Purposeful EDA tied to modeling decisions |
| S-4 | Preprocessing via a reusable sklearn Pipeline |
| S-5 | Class-imbalance handling (cost-sensitive primary; SMOTE optional experiment) |
| S-6 | Stratified train/validation/test methodology |
| S-7 | Training of the three candidate models |
| S-8 | Model comparison under a common protocol |
| S-9 | Evaluation with imbalance-appropriate metrics |
| S-10 | Threshold tuning on validation data |
| S-11 | Final model selection and one-time test evaluation |
| S-12 | Model artifact + threshold config creation |
| S-13 | Integration boundaries for MLflow, DVC, FastAPI, Docker, Pytest |
| S-14 | Secondary Temporal Robustness Experiment: chronological holdout carve-out, one-time evaluation of the frozen primary model/threshold, comparison with primary random-test results |

## 5. Out of Scope

This is a **compact but end-to-end ML/MLOps assessment project**. The following are explicitly excluded:

- Integration with real banking, card-network, or payment-authorization systems
- Real-time streaming infrastructure (e.g., Kafka) and orchestration (e.g., Airflow)
- Kubernetes, Terraform, or complex cloud architecture
- Microservices, feature stores, model registries beyond MLflow's local capabilities
- Large-scale or distributed training
- Deep learning models (not justified for ~30 tabular features and a 2-day window)
- Additional model families beyond the three candidates (see §9)
- Monitoring/drift detection in production, A/B testing, online learning
- Business cost modelling with real monetary figures
- Temporal/rolling-window model retraining, time-series cross-validation, or drift-adaptive models (the temporal experiment is evaluation-only)
- Any second or merged dataset

## 6. Functional Requirements

| ID | Requirement |
|---|---|
| FR-001 | The system shall load the transaction dataset from a configured path and fail with a clear error if the file is absent or unreadable. |
| FR-002 | The system shall validate the schema: presence of all required columns, absence of unexpected columns (reported), and numeric dtypes for all features. |
| FR-003 | The system shall detect and report missing values and non-finite (±inf) values per column. |
| FR-004 | The system shall detect and report exact duplicate rows and apply the documented duplicate policy (DOC-02 §3). |
| FR-005 | The system shall verify that the target column contains only the values {0, 1} and both classes are present. |
| FR-006 | The system shall compute and report class counts and fraud percentage for the full dataset and every split. |
| FR-007 | The system shall split data into stratified train/validation/test sets with a fixed seed before any fitted transformation. |
| FR-008 | The system shall apply preprocessing through a pipeline fitted on training data only. |
| FR-009 | The system shall handle class imbalance using training-data-only techniques (class weighting as primary; SMOTE as optional experiment). |
| FR-010 | The system shall train Logistic Regression, Random Forest, and XGBoost with configuration-driven hyperparameters. |
| FR-011 | The system shall compare candidate models on validation data using the metrics in §10 and produce a comparison table. |
| FR-012 | The system shall compute Precision, Recall, F1, PR-AUC (Average Precision), ROC-AUC, and a Confusion Matrix. |
| FR-013 | The system shall evaluate a range of thresholds on validation probabilities and select one according to the documented objective (§11). |
| FR-014 | The system shall evaluate the frozen model + threshold exactly once on the test set. |
| FR-015 | The system shall persist a single inference-ready artifact (preprocessing + model) and a separate threshold/metadata configuration file. |
| FR-016 | The system shall provide a reproducible inference function that accepts raw feature values in the documented schema and returns probability, class, and threshold. |
| FR-017 | The system shall perform a secondary Temporal Robustness Experiment: order the deduplicated dataset by `Time`, reserve the chronologically latest transactions as a temporal holdout before the primary split, exclude that holdout from all fitting/tuning/selection/threshold decisions, evaluate the frozen primary model and frozen primary threshold on it exactly once, report PR-AUC, ROC-AUC, Precision, Recall, F1, confusion matrix, FP and FN counts, and compare these against the primary random-test results. `Time` shall be used only for ordering and shall not be a model feature. |

## 7. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-001 | Reproducibility | Re-running training with the same data, config, and environment yields the same splits and materially identical metrics. |
| NFR-002 | Randomness control | A single global seed (config) is passed to every stochastic component (split, models, SMOTE, CV). |
| NFR-003 | Maintainability | Code is organized into small modules (data, features, train, evaluate, threshold, inference) with no notebook-only logic in the critical path. |
| NFR-004 | Modularity | Each pipeline stage has explicit inputs/outputs on disk to support DVC stages later. |
| NFR-005 | Explainability | Logistic Regression coefficients and tree-model feature importances are reported; decisions are documented. |
| NFR-006 | Testability | Core functions (validation, splitting, preprocessing, threshold selection, inference) are pure enough to be unit-tested with Pytest. |
| NFR-007 | Local execution | Full training completes on a standard laptop CPU (no GPU required) within a reasonable time (target: under ~30 min end-to-end). **[ASM]** |
| NFR-008 | Dependency clarity | All dependencies are pinned in a single requirements file; no tools outside the agreed stack. |
| NFR-009 | No data leakage | No statistic, resampling, or threshold is derived from validation/test data except where explicitly designated (threshold from validation only). |
| NFR-010 | Schema consistency | Feature names and order are fixed at training, stored in metadata, and enforced at inference. `Time` is excluded from the model feature set (ordering use only). |

## 8. Input and Output Specification (Conceptual)

**Input.** A single transaction containing all model features in the training schema. Under the expected dataset: `V1`–`V28`, `Amount` (29 numeric values). `Time` is not a model input (it is used only to order data for the temporal robustness experiment); `Class` is never an input. **[ASM — verify against CSV]**

**Output.**

| Field | Description |
|---|---|
| Fraud probability | Model estimate of P(Class = 1), in [0, 1] |
| Predicted class | 1 (fraud) / 0 (legitimate), derived from probability ≥ threshold |
| Threshold used | The frozen, validation-selected threshold |

The concrete API contract (field names, validation errors, batch support, versioning) is **deferred** to the later System/API design document.

## 9. ML Requirements

The system shall train and compare exactly these three candidates **[REQ]**:

| # | Model | Role |
|---|---|---|
| 1 | Logistic Regression | Baseline; interpretable linear reference |
| 2 | Random Forest | Nonlinear, tree-based bagging benchmark |
| 3 | XGBoost | Gradient-boosted, high-performance candidate |

Additional models shall not be introduced unless a documented gap (e.g., all three fail the success criteria) justifies it via §15.

## 10. Evaluation Requirements

Required metrics **[REQ]**: Precision, Recall, F1-score, PR-AUC (Average Precision), ROC-AUC, Confusion Matrix. The same metrics, plus explicit FP and FN counts, are required for the temporal holdout (FR-017).

**Accuracy shall not be used as a model-selection metric.** It may be reported for completeness only, with a note on why it is misleading.

- **False Negative (FN)** — fraud missed by the detector. Typically the more damaging error in fraud detection.
- **False Positive (FP)** — legitimate transaction flagged as fraud. Causes customer friction and review workload.
- **Precision–recall tradeoff** — lowering the threshold increases recall and FPs; raising it increases precision and FNs. The operating point is a design decision (§11), not a model property.

## 11. Threshold Optimization Requirement

**[REQ]** The fraud decision threshold shall be selected using **validation data only**; 0.50 shall not be assumed.

The design shall evaluate a range of thresholds and select one per a documented objective:

- **Primary objective [REC]:** among thresholds achieving Recall ≥ R_min on validation, choose the one maximizing Precision.
- **Fallback objective [REC]:** if R_min cannot be justified or is unattainable, maximize **F2** (recall weighted more than precision).

R_min is a configuration value, marked **provisional** until validation results are inspected (see DOC-02 §13). No threshold value is fixed in advance.

## 12. Success Criteria

| ID | Criterion | Measurable check |
|---|---|---|
| SC-01 | Data validated | Validation report produced; all FR-002–FR-006 checks pass or are documented. |
| SC-02 | Leakage-safe split | Three stratified splits saved; class ratios within ±0.02 pp of the development pool (tolerance provisional); temporal holdout saved separately. |
| SC-03 | All three models trained | Each model produces validation probabilities and full metric set. |
| SC-04 | Comparison documented | Validation comparison table with all §10 metrics. |
| SC-05 | Threshold selected | Threshold curve and chosen τ with objective recorded. |
| SC-06 | Beats baseline | Selected model's validation PR-AUC ≥ Logistic Regression's PR-AUC, and far above the positive-class prevalence (the no-skill PR-AUC). |
| SC-07 | Single test evaluation | Test metrics reported once for the frozen configuration. |
| SC-08 | Artifacts persisted | Model artifact + threshold/metadata config load and reproduce test predictions. |
| SC-09 | Reproducible | A second run reproduces splits exactly and metrics within floating-point tolerance. |
| SC-10 | Temporal robustness reported | Temporal holdout evaluated once with the frozen model/threshold; all FR-017 metrics reported; side-by-side comparison with primary test metrics including absolute (and, where meaningful, relative) change; holdout fraud prevalence reported; interpretation recorded without claims of production drift. No minimum temporal performance is mandated. |

No absolute metric targets (e.g., "Recall ≥ 0.90") are mandated; any such target is **provisional** and set only after validation analysis.

## 13. Constraints and Assumptions

| ID | Type | Statement |
|---|---|---|
| C-01 | Constraint | ~2-day implementation window. |
| C-02 | Constraint | Local development on a CPU laptop. |
| C-03 | Constraint | Open/public dataset (ULB/Kaggle Credit Card Fraud Detection). |
| C-04 | Constraint | Binary classification only. |
| C-05 | Constraint | Deployment later via FastAPI + Docker. |
| C-06 | Constraint | MLOps later via DVC + MLflow (local). |
| A-01 | Assumption | Dataset has columns `Time`, `V1`–`V28`, `Amount`, `Class`. **Verify.** |
| A-02 | Assumption | Fraud prevalence is severe (documented ~0.17%). **Verify.** |
| A-03 | Assumption | `V1`–`V28` are anonymized (documented as PCA outputs); no semantic meaning is assumed. |
| A-04 | Assumption | Data fits in memory (~150 MB CSV). **Verify.** |
| A-05 | Assumption | `Time` is a monotone relative timestamp suitable for chronological ordering (documented as seconds since the first transaction). **Verify.** |
| A-06 | Assumption | The latest-time holdout contains enough fraud cases for meaningful (if noisy) metrics. **Verify count after carve-out.** |

**Risks and limitations (temporal robustness)**

| ID | Statement |
|---|---|
| L-01 | The dataset spans only ~2 days; the temporal holdout covers hours, not months. Results indicate possible short-horizon temporal/distribution shift only — **not** proof of real-world production drift. |
| L-02 | Few fraud cases in the temporal holdout make metrics high-variance; small differences vs. the random test may be noise. |
| L-03 | Fraud prevalence may differ between the temporal holdout and the random test, changing the no-skill PR-AUC baseline and Precision; comparisons must report prevalence alongside metrics. |
| L-04 | Reserving the temporal holdout reduces the data available to the primary experiment. |
| L-05 | Removing `Time` as a feature is a deliberate change; any predictive value `Time` had is forgone in exchange for a time-agnostic model. |

## 14. Traceability

| Req ID | Description | Area | Later phase / tool |
|---|---|---|---|
| FR-001 | Dataset loading | Data | DVC (tracked input) |
| FR-002 | Schema validation | Data quality | Pytest |
| FR-003 | Missing/non-finite check | Data quality | Pytest |
| FR-004 | Duplicate check/policy | Data quality | Pytest |
| FR-005 | Target validation | Data quality | Pytest |
| FR-006 | Class distribution | EDA / Data | MLflow (logged stats) |
| FR-007 | Stratified split | ML methodology | DVC stage output |
| FR-008 | Pipeline preprocessing | ML / Serving | FastAPI, Pytest |
| FR-009 | Imbalance handling | ML | MLflow params |
| FR-010 | Train 3 models | ML | MLflow runs |
| FR-011 | Model comparison | ML | MLflow metrics |
| FR-012 | Metric computation | Evaluation | MLflow metrics, Pytest |
| FR-013 | Threshold optimization | Evaluation | MLflow param/artifact |
| FR-014 | Single test evaluation | Evaluation | MLflow metrics |
| FR-015 | Artifact persistence | Engineering | MLflow artifact, Docker image |
| FR-016 | Inference function | Serving | FastAPI, Pytest |
| NFR-001/002 | Reproducibility/seeds | Engineering | Git, DVC, MLflow |
| NFR-008 | Pinned dependencies | Engineering | Docker, CI |
| NFR-009 | No leakage | ML methodology | Pytest |
| NFR-010 | Schema consistency | Serving | FastAPI, Pytest |
| FR-017 | Temporal robustness experiment | Evaluation / ML methodology | DVC stage output, MLflow metrics, Pytest (holdout isolation) |

## 15. Decision Log

| ID | Decision | Status |
|---|---|---|
| D-01 | Problem framed as probabilistic binary classification with a tuned threshold. | Confirmed |
| D-02 | Candidate models fixed to LR, RF, XGBoost. | Confirmed |
| D-03 | Accuracy excluded from model selection. | Confirmed |
| D-04 | Threshold selected on validation only; never on test. | Confirmed |
| D-05 | Threshold objective: Recall ≥ R_min then max Precision; F2 fallback. | Provisional (R_min) |
| D-06 | Test set evaluated exactly once. | Confirmed |
| D-07 | Artifact = one pipeline (preprocessing + model) + separate threshold/metadata file. | Confirmed |
| D-08 | Out-of-scope infrastructure excluded (K8s, Airflow, Kafka, Terraform, cloud, DL). | Confirmed |
| D-09 | Dataset schema assumptions pending CSV inspection. | Requires verification |
| D-10 | Add a secondary Temporal Robustness Experiment; primary random-split protocol unchanged for model comparison, threshold optimization, and primary evaluation. | Confirmed |
| D-11 | `Time` used only for chronological ordering; excluded from model features (supersedes the v1.0 inclusion of `Time` as a feature). | Confirmed |
| D-12 | Temporal holdout is evaluation-only: never used for preprocessing fitting, tuning, model selection, threshold tuning, or retraining. | Confirmed |
| D-13 | Temporal degradation is interpreted as evidence of possible temporal/distribution shift, not proof of production drift. | Confirmed |
