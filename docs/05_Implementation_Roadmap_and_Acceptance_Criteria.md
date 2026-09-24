# Credit Card Fraud Detection — Implementation Roadmap & Acceptance Criteria

| Field | Value |
|---|---|
| Document ID | DOC-05 |
| Status | Baseline v1.0 (Final Source of Truth for implementation) |
| Depends on | DOC-01 v1.1 · DOC-02 v1.1 · DOC-03 v1.0 · DOC-04 v1.0 (all authoritative, unmodified) |
| Tags | **[INHERITED]** fixed by DOC-01–04 · **[EMPIRICAL]** needs experiment results · **[OPTIONAL]** may be skipped without failing the project · **[IMP]** implementation-level choice made here (logged in §23) |
| Change policy | Any deviation from DOC-01–04 discovered during coding must be identified, justified, and logged in §23 before it is implemented. |

**Verified dataset facts** (inspection of `data/raw/creditcard.csv`): 284,807 raw rows · 492 frauds · 31 columns (`Time`, `V1`–`V28`, `Amount`, `Class`) · no missing values · no infinite values · 1,081 exact duplicate rows. Derived by arithmetic: 283,726 rows after deduplication. Fraud count after deduplication, temporal-holdout size/fraud count and split counts are **[EMPIRICAL]** and are produced by the pipeline.

---

## 1. Purpose

This document converts the four design documents into an executable, milestone-by-milestone build plan with objective acceptance criteria. It adds **no** new methodology or architecture.

| Document | Answers |
|---|---|
| DOC-01 | **WHAT** — requirements, scope, success criteria |
| DOC-02 | **ML HOW** — data, validation, splits, leakage, models, metrics, threshold, temporal robustness |
| DOC-03 | **MLOps HOW** — Git, DVC, MLflow, config, lineage, tests, CI |
| DOC-04 | **SERVING HOW** — FastAPI contract, startup checks, Docker, readiness |
| **DOC-05** | **BUILD ORDER + ACCEPTANCE** — sequence, files, commands, checks, Definition of Done |

## 2. Implementation Principles

| # | Principle |
|---|---|
| P-1 | Implement incrementally; one milestone at a time, in §5 order. |
| P-2 | Keep the repository runnable and committed after every milestone. |
| P-3 | Write the unit tests for a boundary in the same milestone that creates it. |
| P-4 | Never violate leakage rules (§10). Structural guards (DVC deps, module imports) beat conventions. |
| P-5 | The temporal holdout is written once by `split` and read only by `evaluate_temporal`. |
| P-6 | Every tunable value lives in `config/config.yaml`; serving values come only from `model_config.json`. |
| P-7 | Preprocessing lives only inside the saved Pipeline; the API does no transformation. |
| P-8 | Preserve reproducibility: seeded, pinned, committed `dvc.lock`. |
| P-9 | Add no dependency outside the agreed stack. |
| P-10 | Do not over-engineer: prefer plain functions, small modules, simple CLIs. |

## 3. Final Repository Structure

```text
.
├── config/
│   └── config.yaml                  # single parameter source (§7) — DVC params
├── data/                            # all git-ignored; DVC-managed
│   ├── raw/creditcard.csv           # immutable raw data (creditcard.csv.dvc in Git)
│   ├── interim/clean.csv            # validated + deduplicated (validate stage)
│   └── processed/                   # train.csv, val.csv, test.csv, temporal_holdout.csv (split stage)
├── docs/                            # DOC-01 … DOC-05
├── src/fraud_detection/
│   ├── __init__.py
│   ├── config.py                    # load + validate config.yaml
│   ├── eda.py                       # descriptive EDA script (not a DVC stage) [IMP]
│   ├── data/
│   │   ├── validate.py              # checks + dedup + SHA-256
│   │   └── split.py                 # temporal carve-out + stratified 70/15/15
│   ├── features/preprocess.py       # ColumnTransformer factory (29 features)
│   ├── models/
│   │   ├── train.py                 # estimator factories, class weights, fit
│   │   ├── tuning.py                # RandomizedSearchCV on train only
│   │   └── select.py                # DOC-02 §12 selection framework
│   ├── evaluation/
│   │   ├── metrics.py               # P, R, F1, F2, PR-AUC, ROC-AUC, CM, FP/FN, prevalence, lift
│   │   ├── threshold.py             # grid + PR points, recall≥R_min→max precision, F2 fallback
│   │   ├── evaluate.py              # primary test evaluation (once)
│   │   └── evaluate_temporal.py     # temporal evaluation + comparison + final model_config
│   ├── tracking/mlflow_utils.py     # run creation/resume, tags, logging helpers
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py             # load/verify artifacts, predict, shared p ≥ τ rule
│   └── pipeline/                    # DVC entrypoints: validate.py, split.py, train.py,
│                                    #   evaluate.py, evaluate_temporal.py
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app, lifespan loading, /health, /predict
│   └── schemas.py                   # Pydantic request/response models
├── models/                          # DVC outputs; git-ignored binaries
│   ├── candidates/{lr,rf,xgb}.joblib
│   ├── model.joblib
│   ├── train_metadata.json
│   └── model_config.json
├── reports/
│   ├── validation_report.json · split_summary.json
│   ├── model_comparison.csv · threshold_analysis_<model>.csv · model_selection.md
│   ├── test_metrics.json
│   ├── eda/                         # ≤ 8 figures + eda_summary.md
│   ├── figures/                     # threshold plots, confusion matrices
│   └── temporal/                    # temporal_split_summary.json, temporal_metrics.json,
│                                    #   temporal_vs_random_comparison.csv,
│                                    #   temporal_confusion_matrix.png, temporal_robustness.md
├── notebooks/eda.ipynb              # [OPTIONAL] descriptive only
├── tests/
│   ├── conftest.py                  # synthetic fixtures, fixture artifact pair
│   ├── test_validation.py · test_split.py · test_leakage.py
│   ├── test_threshold.py · test_inference.py · test_artifacts.py
│   └── test_api.py
├── .github/workflows/ci.yml         # [OPTIONAL]
├── dvc.yaml · dvc.lock · .dvcignore · .dvc/
├── pyproject.toml                   # minimal package metadata for `pip install -e .` [IMP]
├── requirements.txt                 # full pinned set (train + test + serve)
├── requirements-serve.txt           # pinned inference-only set
├── Dockerfile · .dockerignore
├── .gitignore
└── README.md
```

## 4. Milestone Plan

Each milestone ends with a commit (see §16). Commands are indicative (bash); Windows equivalents are acceptable.

### M0 — Repository and Environment Setup

| Item | Detail |
|---|---|
| Objective | Runnable skeleton, pinned environment, Git + DVC initialized, raw data tracked. |
| Prerequisites | Python installed [IMP: record exact version], Git, `creditcard.csv` available locally. |
| Tasks | 1. `git init`; create tree from §3 (empty modules with docstrings). 2. Write `.gitignore` (data/, models/*.joblib, models/candidates/, mlruns/, mlflow.db, .venv/, caches, `*.png` under reports if DVC-managed). 3. Create venv; write `requirements.txt` with exact pins [EMPIRICAL: versions chosen at install]; write `requirements-serve.txt` with identical pins for shared libs. 4. Minimal `pyproject.toml` (package `fraud_detection` under `src/`). 5. `pip install -r requirements.txt && pip install -e .`. 6. `dvc init`; copy CSV to `data/raw/creditcard.csv`; `dvc add data/raw/creditcard.csv`. 7. Configure local remote; `dvc push`. 8. Copy DOC-01–05 into `docs/`. |
| Files | `.gitignore`, `requirements*.txt`, `pyproject.toml`, `src/fraud_detection/**/__init__.py`, `data/raw/creditcard.csv.dvc`, `.dvc/config`, `docs/` |
| Inputs | Raw CSV |
| Outputs | Initialized repo; DVC-tracked raw data |
| Commands | `dvc init` · `dvc add data/raw/creditcard.csv` · `dvc remote add -d localstore ../dvc-storage` · `dvc push` · `python -c "import fraud_detection, sklearn, xgboost, mlflow, fastapi"` |
| Tests | Import smoke only |
| Acceptance | `git status` clean after commit; `data/raw/creditcard.csv` absent from Git, `.dvc` file present; `dvc status` clean; all imports succeed. |
| Failure modes | CSV accidentally committed to Git; unpinned requirements; missing `__init__.py`. |
| Must NOT | Commit the CSV; add any library outside §1 stack of DOC-03/§16 of DOC-02. |
| Checkpoint | Commit `chore: repository skeleton, pinned env, DVC-tracked raw data`. |

### M1 — Dataset Validation and Data Preparation

| Item | Detail |
|---|---|
| Objective | Implement DOC-02 §3 checks and deduplication; produce `clean.csv` and validation report. |
| Prerequisites | M0; `config.yaml` (schema/paths/seed sections — created here). |
| Tasks | 1. Write `config/config.yaml` §7 sections needed now; `config.py` loader with validation (required keys, `Time` ∉ features, 29 features, fractions sum). 2. `data/validate.py`: file existence, required columns, unexpected columns (warn), dtypes numeric, missing (hard fail on `Class`), ±inf (hard fail), `Class` ⊆ {0,1} with both present, negative `Amount`/`Time` (hard fail), outlier report (no removal), duplicate removal (keep first) with per-class counts, SHA-256 of raw file. 3. `pipeline/validate.py` CLI writing `data/interim/clean.csv` and `reports/validation_report.json`. 4. Unit tests. |
| Files | `config/config.yaml`, `src/fraud_detection/config.py`, `data/validate.py`, `pipeline/validate.py`, `tests/conftest.py`, `tests/test_validation.py` |
| Inputs | `data/raw/creditcard.csv` |
| Outputs | `data/interim/clean.csv`, `reports/validation_report.json` |
| Commands | `python -m fraud_detection.pipeline.validate` · `pytest tests/test_validation.py` |
| Tests | Missing file, missing column, unexpected column, non-numeric, inf, bad target, single class, negative Amount/Time, duplicates removed |
| Acceptance | Report shows 284,807 raw rows, 492 raw frauds, 1,081 duplicates removed, 283,726 clean rows, zero missing/inf, SHA-256 present; clean fraud count recorded [EMPIRICAL]; tests pass. |
| Failure modes | Duplicate count ≠ 1,081 (dtype/float parsing changed values); reading with index column. |
| Must NOT | Remove outliers; impute `Class`; modify the raw file. |
| Checkpoint | Commit `data: validation and deduplication`. |

### M2 — EDA

| Item | Detail |
|---|---|
| Objective | Descriptive EDA per DOC-02 §4 on the deduplicated data. |
| Prerequisites | M1 (`clean.csv`). |
| Tasks | `eda.py` producing the DOC-02 §4 analyses (shape, missing/duplicates, class counts/log bar, describe, Amount by class, Time density + fraud rate over time, V-feature mean differences, top-separating KDE grid, correlation with Class, correlation heatmap) — **≤ 8 figures** — and `reports/eda/eda_summary.md` linking findings to decisions. Notebook [OPTIONAL]. |
| Files | `src/fraud_detection/eda.py`, `reports/eda/*`, [OPTIONAL] `notebooks/eda.ipynb` |
| Inputs | `data/interim/clean.csv` |
| Outputs | `reports/eda/` |
| Commands | `python -m fraud_detection.eda` |
| Tests | None (descriptive) |
| Acceptance | ≤ 8 figures; summary written with no placeholder text; no file under `data/processed/` or `models/` created or read. |
| Failure modes | Plot overload; fitting anything during EDA. |
| Must NOT | Use EDA output as a pipeline input; derive features from `Time`; make EDA a DVC stage. |
| Checkpoint | Commit `docs: EDA report`. |

### M3 — Leakage-Safe Splitting and Preprocessing

| Item | Detail |
|---|---|
| Objective | Temporal carve-out, stratified primary split, preprocessing factory. |
| Prerequisites | M1. |
| Tasks | 1. `data/split.py`: stable sort by `Time` (`kind="mergesort"`), `cut_index = floor(0.85·n)`, move boundary-tie rows into holdout, write `temporal_holdout.csv`; development pool → stratified 70/30 then 50/50 (seed 42); write `train/val/test.csv`; `split_summary.json` (counts, prevalence per split) and `reports/temporal/temporal_split_summary.json` (counts, fraud count, prevalence, Time min/max for pool and holdout, low-confidence flag if frauds < `min_fraud_warning`). 2. `pipeline/split.py` CLI. 3. `features/preprocess.py`: ColumnTransformer — `V1`–`V28` median impute → StandardScaler; `Amount` median impute → RobustScaler; `Time` never passed. 4. Tests. |
| Files | `data/split.py`, `pipeline/split.py`, `features/preprocess.py`, `tests/test_split.py`, `tests/test_leakage.py` (preprocessor part) |
| Inputs | `data/interim/clean.csv`, config |
| Outputs | `data/processed/{train,val,test,temporal_holdout}.csv`, split summaries |
| Commands | `python -m fraud_detection.pipeline.split` · `pytest tests/test_split.py tests/test_leakage.py` |
| Tests | Ratios; stratification tolerance (DOC-01 SC-02); `max(Time_pool) ≤ min(Time_holdout)`; tie handling; zero row overlap across all four partitions; preprocessor fitted stats equal train stats; `Time` not in transformer columns |
| Acceptance | Four files exist; overlap = 0; ordering invariant holds; holdout fraud count recorded [EMPIRICAL]; tests pass. |
| Failure modes | Row identity lost (use a stable row id column internally for overlap tests, dropped before modelling [IMP]); stratifying the holdout. |
| Must NOT | Stratify or shuffle the temporal holdout; fit anything before splitting; include `Time` in features. |
| Checkpoint | Commit `ml: temporal carve-out, stratified split, preprocessing`. Tag `v0.1-data` [INHERITED]. |

### M4 — Baseline and Candidate Model Training

| Item | Detail |
|---|---|
| Objective | Fit LR (baseline), RF, XGB with cost-sensitive imbalance handling and default/fixed params. |
| Prerequisites | M3. |
| Tasks | `models/train.py`: estimator factories per DOC-02 §9 (LR `class_weight="balanced"`, lbfgs, max_iter 1000; RF `class_weight="balanced_subsample"`; XGB `scale_pos_weight = n_neg_train / n_pos_train`, `binary:logistic`, `aucpr`, `hist`); each wrapped as `Pipeline([("preprocess", …), ("model", …)])`. `evaluation/metrics.py`: all metric functions. Quick validation sanity metrics (PR-AUC) to confirm training works. |
| Files | `models/train.py`, `evaluation/metrics.py` |
| Inputs | `train.csv`, `val.csv` |
| Outputs | In-memory fitted pipelines (persisted in M5/M6) |
| Commands | Run via `pipeline/train.py` in development mode |
| Tests | Unit test: metric functions on hand-computed toy arrays (in `test_threshold.py` or `test_inference.py` fixtures) |
| Acceptance | All three pipelines fit and produce probabilities on validation; PR-AUC > validation prevalence for each [EMPIRICAL]. |
| Failure modes | `scale_pos_weight` computed on full data; LR convergence warnings (increase nothing but `max_iter` per config). |
| Must NOT | Add models; read `test.csv`/`temporal_holdout.csv`. |
| Checkpoint | Commit `ml: candidate model training`. |

### M5 — Hyperparameter Tuning and MLflow Tracking

| Item | Detail |
|---|---|
| Objective | Train-only CV tuning; one MLflow candidate run per model. |
| Prerequisites | M4. |
| Tasks | 1. `models/tuning.py`: `RandomizedSearchCV` (LR small grid), `StratifiedKFold(3, shuffle, seed)`, `scoring="average_precision"`, `n_iter ≤ 10`, spaces from config; refit best on full train. 2. `tracking/mlflow_utils.py`: set URI/experiment from config; tag helper (`git_commit`, `git_dirty`, `data_sha256`, `data_dvc_md5`, `config_hash`, `stage`, `run_type`). 3. Log candidate runs (§9). 4. [OPTIONAL] SMOTE+LR via imblearn Pipeline if `imbalance.smote_experiment: true`. |
| Files | `models/tuning.py`, `tracking/mlflow_utils.py` |
| Inputs | `train.csv` (CV), `val.csv` (metrics) |
| Outputs | `models/candidates/{lr,rf,xgb}.joblib`, MLflow candidate runs |
| Commands | `mlflow ui` (inspection) |
| Tests | `test_leakage.py`: SMOTE (if enabled) only increases train size; CV splitter receives train only |
| Acceptance | 3 candidate runs (+1 if SMOTE run) with params, `val_*` metrics, artifacts; best params persisted. |
| Failure modes | RF search too slow → reduce `n_iter` to 5 in config and log it (DOC-02 §10). |
| Must NOT | Tune on validation; use MLflow Model Registry; enlarge search spaces. |
| Checkpoint | Commit `mlops: tuning and MLflow candidate tracking`. |

### M6 — Model Comparison and Threshold Optimization

| Item | Detail |
|---|---|
| Objective | Validation threshold analysis per model, selection per DOC-02 §12, freeze. |
| Prerequisites | M5. |
| Tasks | 1. `evaluation/threshold.py`: grid 0.01–0.99 step 0.01 ∪ `precision_recall_curve` points; per τ: P, R, F1, F2, FP, FN; rule recall ≥ `r_min` → max precision (tie → higher τ); fallback max F2; record objective used. 2. `models/select.py`: eligibility → PR-AUC → operating point (precision at required recall / F2, FP count) → tie-break (PR-AUC within 0.01 → fewer FP → simpler) → LR-baseline sanity. 3. `pipeline/train.py`: orchestrate M4–M6; write `model_comparison.csv`, `threshold_analysis_<model>.csv`, P/R-vs-τ plots, `model_selection.md`, `models/model.joblib`, `models/train_metadata.json` (model name, version, τ, objective, `r_min`, feature_order, seed, data hashes, library versions, validation metrics, final MLflow `run_id`). 4. Create final MLflow run here. |
| Files | `evaluation/threshold.py`, `models/select.py`, `pipeline/train.py`, `tests/test_threshold.py` |
| Inputs | `train.csv`, `val.csv` only |
| Outputs | Train-stage outputs (DOC-03 §4.2 row 3) |
| Commands | `python -m fraud_detection.pipeline.train` · `pytest tests/test_threshold.py` |
| Tests | Known curve → expected τ; unattainable R_min → F2 fallback; tie → higher τ; selection tie-break logic on synthetic metric tables |
| Acceptance | `model_selection.md` lists every candidate's numbers and the rule-based choice; τ and objective recorded; `train_metadata.json` complete; `r_min` confirmed or revised in config with reason logged in §23 before any test evaluation [EMPIRICAL]. |
| Failure modes | τ = 0.99 or 0.01 edges (report; do not force); no model meets R_min (fallback applies automatically). |
| Must NOT | Pick the winner manually; look at test/temporal data; revise `r_min` after M7. |
| Checkpoint | Commit `ml: model selection and validation threshold`. |

### M7 — Final Test + Temporal Robustness Evaluation

| Item | Detail |
|---|---|
| Objective | One primary test evaluation, then one temporal evaluation, with frozen artifacts. |
| Prerequisites | M6 frozen (`model.joblib`, `train_metadata.json` committed). |
| Tasks | 1. `evaluation/evaluate.py` + `pipeline/evaluate.py`: load artifacts from disk, predict `test.csv`, metrics at τ and 0.50 (reference), prevalence, PR-AUC lift; write `test_metrics.json`, `figures/test_confusion_matrix.png`; resume final MLflow run and log `test_*`. 2. `evaluation/evaluate_temporal.py` + `pipeline/evaluate_temporal.py`: same on `temporal_holdout.csv`; comparison table (DOC-02 §21.5) with absolute and relative Δ; [OPTIONAL] seeded bootstrap CIs; `temporal_robustness.md` with interpretation rules (DOC-02 §21.6); assemble final `models/model_config.json` (DOC-02 §15 + DOC-03 §7 lineage keys); log `temporal_*`, `delta_*`, artifacts; end run. |
| Files | `evaluation/evaluate.py`, `evaluation/evaluate_temporal.py`, `pipeline/evaluate.py`, `pipeline/evaluate_temporal.py` |
| Inputs | Frozen artifacts; `test.csv`; then `temporal_holdout.csv` |
| Outputs | `reports/test_metrics.json`, test CM, all `reports/temporal/*`, `models/model_config.json` |
| Commands | `python -m fraud_detection.pipeline.evaluate` then `python -m fraud_detection.pipeline.evaluate_temporal` |
| Tests | `test_artifacts.py`: round-trip; `model_config.json` τ == `train_metadata.json` τ |
| Acceptance | Each command run once for the release; τ unchanged; comparison table fully populated [EMPIRICAL]; interpretation says "possible short-horizon temporal/distribution shift", never "production drift". |
| Failure modes | Re-running evaluation "to see if it improves" after code changes to models (forbidden). |
| Must NOT | Retrain, reselect or retune after seeing results; merge val into train; tune τ on test or holdout. |
| Checkpoint | Commit `ml: final test and temporal robustness evaluation`. (`v1.0-model` is tagged after M8 — IMP-04.) |

### M8 — DVC Pipeline and Reproducibility

| Item | Detail |
|---|---|
| Objective | Encode the five stages in `dvc.yaml` and prove reproduction. |
| Prerequisites | M1–M7 entrypoints working. |
| Tasks | Write `dvc.yaml` per §8; `dvc repro`; declare metrics/plots; commit `dvc.lock`; `dvc push`; reproduction check from a fresh clone. |
| Files | `dvc.yaml`, `dvc.lock`, `.dvcignore` |
| Commands | `dvc dag` · `dvc repro` · `dvc status` · `dvc metrics show` · `dvc push` |
| Tests | Full `pytest` |
| Acceptance | DOC-03 MAC-01…MAC-04 pass; `dvc dag` shows no edge from `test.csv`/`temporal_holdout.csv` into `train`, nor from `temporal_holdout.csv` into `evaluate`. |
| Failure modes | Output declared by two stages; `model_config.json` written by `train` (must not be). |
| Must NOT | Add stages (EDA stays outside); cache raw data as a stage output. |
| Checkpoint | Commit `mlops: DVC pipeline and lock`. Tag `v1.0-model` [INHERITED, timing per IMP-04]. |

**Note on M7/M8 ordering [IMP]:** the one-time test/temporal evaluation of record is the one produced by `dvc repro` in M8. Development runs in M7 validate the code; if M8 regenerates identical results (seeded), no additional "look" at test data has influenced any decision, because model, τ and selection were frozen in M6 and nothing upstream changes between M7 and M8.

### M9 — FastAPI Model Serving

| Item | Detail |
|---|---|
| Objective | DOC-04 API: `/health`, `/predict`, fail-fast startup. |
| Prerequisites | Final `models/model.joblib` + `models/model_config.json`. |
| Tasks | `inference/predictor.py`: load + 8 startup checks (DOC-04 §4), DataFrame in `feature_order`, `predict_proba[:,1]`, shared `apply_threshold(p, τ)` (`≥`). `app/schemas.py`: request model with 29 strict finite float fields (`extra="forbid"`, `allow_inf_nan=False`, strict numeric, `Amount ≥ 0`), response model. `app/main.py`: lifespan loading via env vars `MODEL_PATH`, `MODEL_CONFIG_PATH`; endpoints; generic 500 handler; no payload logging. Refactor `evaluate*.py` to use `apply_threshold`. |
| Files | `inference/predictor.py`, `app/main.py`, `app/schemas.py`, `tests/test_inference.py`, `tests/test_api.py` |
| Commands | `uvicorn app.main:app --reload --port 8000` · `pytest tests/test_inference.py tests/test_api.py` |
| Acceptance | DOC-04 DR-02, DR-03, DR-04 pass. |
| Failure modes | `app` importing training modules; Pydantic coercing `"1.2"` strings. |
| Must NOT | Retrain or rewrite artifacts at startup; accept `Time`; add endpoints, auth, frontend. |
| Checkpoint | Commit `serve: FastAPI prediction service`. |

### M10 — Dockerization

| Item | Detail |
|---|---|
| Objective | Serving image per DOC-04 §8. |
| Prerequisites | M9; `models/` populated. |
| Tasks | Write `Dockerfile` and `.dockerignore` per §14; build; run; compare with local. |
| Files | `Dockerfile`, `.dockerignore` |
| Commands | see §14 |
| Acceptance | DOC-04 DR-05, DR-06, DR-08 pass. |
| Failure modes | Missing `models/` at build; version drift between requirement files. |
| Must NOT | Copy `data/`, `mlruns/`, training modules; run DVC or training in the build. |
| Checkpoint | Commit `docker: serving image`. `v1.0-serve` is tagged in M12. |

### M11 — Pytest and Optional CI

| Item | Detail |
|---|---|
| Objective | Complete and green test suite; optional CI. |
| Prerequisites | M1–M10. |
| Tasks | Fill gaps against §15; register `artifact` marker in `pyproject.toml`; fixture artifact pair in `conftest.py`; [OPTIONAL] `.github/workflows/ci.yml` (checkout → setup-python → install → [OPTIONAL] ruff → `pytest -m "not artifact"`). |
| Commands | `pytest` · `pytest -m "not artifact"` |
| Acceptance | All tests pass locally; CI green if enabled. |
| Must NOT | Weaken tests or methodology to get green; add DVC/Docker/deploy steps to CI. |
| Checkpoint | Commit `test: complete suite` / `ci: lightweight workflow`. |

### M12 — Final Validation, README and Submission

| Item | Detail |
|---|---|
| Objective | Documentation and final audit (§22). |
| Tasks | README: overview, architecture diagram, setup, reproduce (`dvc pull && dvc repro`), MLflow UI, API usage + example request, Docker, tests, results summary copied from generated reports [EMPIRICAL], temporal findings with limitations, link to docs. Run §22 audit; fresh-clone reproduction. |
| Acceptance | §19 Definition of Done satisfied. |
| Checkpoint | Commit `docs: README and final audit`; tag `v1.0-serve`. |

## 5. Detailed Implementation Order

Corrected against DOC-02/03/04 (changes from the naive order are marked ◆):

```text
Environment + repository skeleton + pyproject
 ↓
DVC init + dvc add raw data + local remote            ◆ raw data tracked before any processing (DOC-03 §4.1)
 ↓
config.yaml + config loader
 ↓
Data validation + deduplication (validate stage code)
 ↓
EDA on deduplicated data (descriptive only)           ◆ DOC-02 §4: EDA uses full deduplicated data, before splits
 ↓
Temporal carve-out → primary stratified split (split stage code)
 ↓
Preprocessing factory
 ↓
Metrics module
 ↓
LR baseline → RF → XGB (cost-sensitive)
 ↓
Train-only CV tuning + MLflow candidate runs
 ↓
Threshold analysis (validation) → model selection → freeze τ
   → model.joblib + train_metadata.json + final MLflow run created
 ↓
Primary test evaluation (once)
 ↓
Temporal robustness evaluation (once) → final model_config.json
 ↓
dvc.yaml + dvc repro + dvc.lock (reproducibility of record)
 ↓
Inference predictor + shared decision rule
 ↓
FastAPI
 ↓
Docker
 ↓
Test-suite completion (unit tests written incrementally from M1) ◆
 ↓
[OPTIONAL] CI
 ↓
README + final audit
```

## 6. File-Level Implementation Map

| File | Purpose | Milestone | Depends on | Acceptance check |
|---|---|---|---|---|
| `config/config.yaml` | All parameters (§7) | M1 (extended M3–M7) | — | Loader validation passes; 29 features; no `Time` |
| `src/fraud_detection/config.py` | Load + validate config | M1 | config.yaml | Invalid config raises |
| `data/validate.py` | DOC-02 §3 checks, dedup, SHA-256 | M1 | config.py | `test_validation.py` green |
| `pipeline/validate.py` | Stage entrypoint | M1 | validate.py | Writes clean.csv + report |
| `eda.py` | EDA figures + summary | M2 | clean.csv | ≤ 8 figures |
| `data/split.py` | Carve-out + stratified split | M3 | config.py | `test_split.py` green |
| `pipeline/split.py` | Stage entrypoint | M3 | split.py | 4 CSVs + 2 summaries |
| `features/preprocess.py` | ColumnTransformer | M3 | config.py | Fit-on-train test green |
| `evaluation/metrics.py` | All metrics | M4 | — | Toy-array tests green |
| `models/train.py` | Estimator + Pipeline factories | M4 | preprocess.py | 3 pipelines fit |
| `models/tuning.py` | Train-only RandomizedSearchCV | M5 | train.py | Search uses train only |
| `tracking/mlflow_utils.py` | Runs, tags, resume | M5 | config.py | Candidate + final runs visible |
| `evaluation/threshold.py` | Threshold search + rule | M6 | metrics.py | `test_threshold.py` green |
| `models/select.py` | DOC-02 §12 framework | M6 | metrics, threshold | Deterministic selection |
| `pipeline/train.py` | Train stage orchestration | M6 | all above | Train outputs complete |
| `evaluation/evaluate.py` + `pipeline/evaluate.py` | Test evaluation | M7 | predictor rule, metrics | `test_metrics.json` |
| `evaluation/evaluate_temporal.py` + `pipeline/evaluate_temporal.py` | Temporal evaluation + final config | M7 | evaluate outputs | `reports/temporal/*`, `model_config.json` |
| `inference/predictor.py` | FR-016 + shared `≥ τ` rule | M7 (rule) / M9 (loader) | none from training | `test_inference.py` green |
| `dvc.yaml` / `dvc.lock` | Pipeline definition / lock | M8 | entrypoints | MAC-01…04 |
| `app/schemas.py` | Pydantic models | M9 | — | Validation tests green |
| `app/main.py` | FastAPI app | M9 | predictor, schemas | `test_api.py` green |
| `tests/*` | See §15 | M1–M11 | modules | `pytest` green |
| `requirements.txt` / `requirements-serve.txt` | Pinned deps | M0 (serve finalized M10) | — | MAC-09 |
| `pyproject.toml` | Package metadata + pytest marker | M0 / M11 | — | `pip install -e .` works |
| `Dockerfile` / `.dockerignore` | Serving image | M10 | M9, models/ | DR-05, DR-06 |
| `.github/workflows/ci.yml` | [OPTIONAL] CI | M11 | tests | Green run |
| `README.md` | Project documentation | M12 | all | §22 documentation audit |

## 7. Configuration Specification

`config/config.yaml` [INHERITED DOC-03 §5]:

```yaml
project:
  name: credit-card-fraud-detection
  model_version: "1.0.0"          # base; final = <this>+<git short sha>
  seed: 42
paths:
  raw: data/raw/creditcard.csv
  interim: data/interim/clean.csv
  processed_dir: data/processed
  models_dir: models
  reports_dir: reports
schema:
  target: Class
  positive_label: 1
  ordering_column: Time
  features: [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14,
             V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount]
  required_columns: [Time, V1, …, V28, Amount, Class]   # written out in full
split:
  train: 0.70
  val: 0.15
  test: 0.15
  stratify: true
temporal:
  holdout_fraction: 0.15
  min_fraud_warning: 20            # provisional
imbalance:
  strategy: class_weight
  smote_experiment: false          # [OPTIONAL]
models:
  enabled: [logistic_regression, random_forest, xgboost]
  logistic_regression: {solver: lbfgs, max_iter: 1000, class_weight: balanced}
  random_forest: {class_weight: balanced_subsample, n_jobs: -1}
  xgboost: {objective: "binary:logistic", eval_metric: aucpr, tree_method: hist, n_jobs: -1}
tuning:
  cv_folds: 3
  n_iter: 10
  scoring: average_precision
  search_spaces: {…per DOC-02 §10…}
threshold:
  r_min: 0.80                      # provisional until M6 validation review
  grid: {start: 0.01, stop: 0.99, step: 0.01}
  include_pr_curve_points: true
  fallback: f2
  reference_threshold: 0.50        # reported only
selection:
  pr_auc_tie_tolerance: 0.01       # provisional
evaluation:
  bootstrap: {enabled: false, n: 1000}   # [OPTIONAL]
mlflow:
  tracking_uri: file:./mlruns
  experiment_name: credit-card-fraud-detection
```

| Configurable | Invariant (asserted in code, not configurable) |
|---|---|
| Seed, paths, fractions, search spaces, fixed model params, imbalance switch, `r_min`, grid, tie tolerance, bootstrap, MLflow URI | `Time` ∉ features; exactly 29 features; `Class` never a feature; temporal holdout not stratified; decision rule `p ≥ τ`; threshold chosen on validation only; test/holdout never read by `train`; three candidate models only |

Serving never reads `config.yaml`; it reads only `model_config.json` [INHERITED DOC-04 DD-08].

## 8. DVC Implementation Sequence

| Step | When | Command / action |
|---|---|---|
| 1 | M0 | `dvc init`; commit `.dvc/`, `.dvcignore` |
| 2 | M0 | `dvc add data/raw/creditcard.csv`; commit `.dvc` file + updated `.gitignore` |
| 3 | M0 | `dvc remote add -d localstore ../dvc-storage`; `dvc push` |
| 4 | M8 | Write `dvc.yaml` with 5 stages (`cmd: python -m fraud_detection.pipeline.<stage>`), `deps`, `params: [config/config.yaml: …]`, `outs`, `metrics` (`cache: false` for `*metrics.json`, `validation_report.json`, `split_summary.json`), `plots` (threshold CSVs) |
| 5 | M8 | `dvc dag` — visually verify dependency rules below |
| 6 | M8 | `dvc repro` |
| 7 | M8 | `dvc status` (must be clean) · `dvc metrics show` |
| 8 | M8 | `git add dvc.yaml dvc.lock reports/*.json …` + commit together with code/config; `dvc push` |
| 9 | M12 | Fresh clone → `dvc pull` → `dvc repro` → `dvc status` clean, `dvc metrics diff` empty/within tolerance |

Stage dependency rules [INHERITED DOC-03 §4.2]:

| Stage | Must depend on | Must NOT depend on |
|---|---|---|
| `validate` | raw CSV, `validate.py`, `pipeline/validate.py` | anything in `data/processed/` |
| `split` | `clean.csv`, `split.py` | models, reports of later stages |
| `train` | `train.csv`, `val.csv`, `features/`, `models/`, `evaluation/threshold.py`, `evaluation/metrics.py`, `tracking/` | **`test.csv`, `temporal_holdout.csv`** |
| `evaluate` | `model.joblib`, `train_metadata.json`, `test.csv` | **`temporal_holdout.csv`** |
| `evaluate_temporal` | `model.joblib`, `train_metadata.json`, `temporal_holdout.csv`, `test_metrics.json` (forces order after `evaluate`) | — |

`models/model_config.json` is an output of `evaluate_temporal` **only**.

## 9. MLflow Implementation Sequence

| When | Action |
|---|---|
| Start of `train` | `mlflow.set_tracking_uri`, `set_experiment` from config |
| Per candidate (M5) | `start_run(run_name=<model>, tags={run_type: candidate, …lineage})` → log → end |
| After selection (M6) | `start_run(run_name="final", tags={run_type: final, …})`; log selection params/val metrics/train artifacts; write `run_id` into `train_metadata.json`; end run |
| `evaluate` (M7) | `start_run(run_id=<from train_metadata>)` → log `test_*`, test artifacts → end |
| `evaluate_temporal` (M7) | resume same run → log `temporal_*`, `delta_*`, temporal artifacts, `model.joblib`, `model_config.json`, `config.yaml` snapshot, `requirements.txt` → end |

**Candidate runs log:** params `model`, best hyperparameters, `cv_folds`, `n_iter`, `seed`, `imbalance_strategy`, `scale_pos_weight` (XGB), `n_features=29`, data identifiers; metrics `val_pr_auc`, `val_roc_auc`, `cv_best_average_precision`, at tuned τ `val_precision/recall/f1/f2/fp/fn/threshold`, at 0.50 `val_ref050_*`; artifacts threshold CSV, P/R-vs-τ plot, validation CM, candidate `.joblib`.

**Final run logs:** `selected_model`, selected hyperparameters, `threshold`, `threshold_objective`, `r_min`, `seed`, `model_version`, `candidate_run_ids`; `val_*`, `test_*`, `temporal_*` (incl. prevalence, PR-AUC lift), `delta_*`; artifacts listed in DOC-03 §6.2; tags `git_commit`, `git_dirty`, `data_sha256`, `data_dvc_md5`, `config_hash`.

**No MLflow Model Registry** [INHERITED MD-09]. MLflow is history; `models/` (DVC) is the served artifact [INHERITED MD-10].

## 10. ML Validation and Leakage Checklist

| # | Rule | Enforced by |
|---|---|---|
| L-01 | Deduplicate before any split | `validate` stage ordering; test |
| L-02 | Temporal holdout carved out before primary split | `split.py` order; test |
| L-03 | Preprocessing fitted only on train | Pipeline + test |
| L-04 | CV only on train | `tuning.py` signature receives train only; test |
| L-05 | SMOTE only inside training folds (imblearn Pipeline) [OPTIONAL path] | test |
| L-06 | `scale_pos_weight` from train counts only | code review + unit test |
| L-07 | Model selection on validation only | `train` stage deps |
| L-08 | Threshold selection on validation only | `train` stage deps; τ copied verbatim afterwards |
| L-09 | Primary test used exactly once (release run) | stage design; M7/M8 note |
| L-10 | Temporal holdout used exactly once | only `evaluate_temporal` reads it |
| L-11 | No retraining/reselection/retuning after test/temporal results | freeze in M6; §23 log |
| L-12 | `Time` absent from features everywhere | config validation, preprocess test, API schema test |
| L-13 | Zero row overlap across train/val/test/holdout | `test_split.py` |
| L-14 | Validation set not merged into train for final model | `train.py`; DOC-02 TD-14 |

## 11. Model Training and Selection Checklist

**Models [INHERITED]:** Logistic Regression (baseline), Random Forest, XGBoost. [OPTIONAL] SMOTE+LR variant.

**Metrics per candidate on validation:** PR-AUC, ROC-AUC, Precision, Recall, F1, F2, FP, FN, confusion matrix — at tuned τ and at 0.50 (reference).

**Threshold [INHERITED]:**
- [ ] Candidates: 0.01–0.99 step 0.01 ∪ PR-curve thresholds
- [ ] Primary: Recall ≥ `r_min` → maximize Precision; tie → higher τ
- [ ] Fallback: maximize F2 when no τ reaches `r_min`
- [ ] `r_min = 0.80` reviewed after validation curves [EMPIRICAL]; any change logged in §23 **before** M7
- [ ] τ and objective written to `train_metadata.json`; never recomputed

**Selection:** DOC-02 §12 framework, executed in code. The winning model and τ are **[EMPIRICAL]**; this roadmap predicts neither.

## 12. Temporal Robustness Implementation Checklist

| # | Requirement | Where |
|---|---|---|
| 1 | Validate + deduplicate | `validate` |
| 2 | Stable sort by `Time` (mergesort) | `split.py` |
| 3 | Reserve latest 15% (`floor(0.85·n)` cut; boundary ties → holdout) | `split.py` |
| 4 | Write `temporal_holdout.csv`; never read again until step 9 | `split` / `evaluate_temporal` |
| 5 | Stratified 70/15/15 on earlier 85% pool | `split.py` |
| 6 | Train, tune, select, threshold normally | `train` |
| 7 | Freeze model + τ | `train_metadata.json` |
| 8 | Evaluate primary test once | `evaluate` |
| 9 | Evaluate temporal holdout once with same artifacts and τ | `evaluate_temporal` |
| 10 | Compare: PR-AUC, ROC-AUC, P, R, F1 @ τ; FP/FN counts and rates; prevalence; PR-AUC lift; absolute + relative Δ; confusion matrices | `evaluate_temporal` |
| 11 | Report with interpretation rules (possible short-horizon shift; fraud-case terms; inconclusive if within noise; low-confidence flag if frauds < 20) | `temporal_robustness.md` |
| 12 | No feedback into training/selection/τ | stage graph |

Expected outputs [INHERITED DOC-02 §21.7]: `data/processed/temporal_holdout.csv`, `reports/temporal/temporal_split_summary.json`, `reports/temporal/temporal_metrics.json`, `reports/temporal/temporal_vs_random_comparison.csv`, `reports/temporal/temporal_confusion_matrix.png`, `reports/temporal/temporal_robustness.md`, `temporal_holdout_metrics` in `model_config.json`, `temporal_*` in the final MLflow run.

## 13. FastAPI Implementation Checklist

**`GET /health`** — [ ] 200 · [ ] `status: "ok"` · [ ] `model_loaded: true` · [ ] `model_name`, `model_version` from config

**`POST /predict`**
- [ ] Exactly 29 required numeric fields `V1`–`V28`, `Amount`
- [ ] `Time` → 422 · `Class` → 422 · unknown extras → 422 (`extra="forbid"`)
- [ ] Missing field → 422 · string/bool/null → 422 · NaN/±Infinity → 422 · `Amount < 0` → 422
- [ ] Response: `fraud_probability`, `prediction`, `label`, `threshold`, `model_name`, `model_version`
- [ ] Unexpected error → 500 `{"detail": "Prediction failed"}`; traceback server-side only; payload not logged

**Startup (lifespan)**
- [ ] Paths from `MODEL_PATH` / `MODEL_CONFIG_PATH` (defaults in `models/`)
- [ ] DOC-04 §4 checks 1–8 in order; any failure → clear log + non-zero exit
- [ ] No retraining, refitting or artifact writes

## 14. Docker Implementation Checklist

- [ ] Base: `python:<X.Y>-slim` where X.Y = training Python version (`model_config.json` `python_version`) [EMPIRICAL exact tag]
- [ ] `WORKDIR /app`; `pip install --no-cache-dir -r requirements-serve.txt` (add `imbalanced-learn` only if the selected model is the SMOTE pipeline)
- [ ] Copy `app/`, `src/fraud_detection/__init__.py`, `src/fraud_detection/inference/`; `ENV PYTHONPATH=/app/src`
- [ ] Copy only `models/model.joblib`, `models/model_config.json`
- [ ] Non-root user; `EXPOSE 8000`; `CMD uvicorn app.main:app --host 0.0.0.0 --port 8000`
- [ ] [OPTIONAL] `HEALTHCHECK` via Python stdlib request to `/health`
- [ ] `.dockerignore`: `data/`, `mlruns/`, `mlflow.db`, `reports/`, `notebooks/`, `tests/`, `models/candidates/`, `models/train_metadata.json`, `.dvc/`, `.git/`, `.venv/`, `src/fraud_detection/{data,features,models,evaluation,tracking,pipeline}/`, `src/fraud_detection/eda.py`

**Important [IMP]:** `src/fraud_detection/__init__.py` must stay empty of imports so the package imports inside the image without training modules.

Planning-level commands:
```bash
dvc pull                                    # ensure models/ present
docker build -t fraud-detector:<model_version> .
docker run --rm -p 8000:8000 fraud-detector:<model_version>
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @sample_request.json
docker run --rm fraud-detector:<model_version> ls /app        # verify contents (DR-06)
```

## 15. Testing Plan

### Unit tests (synthetic fixtures; run in CI)

| File | Tests | Milestone |
|---|---|---|
| `test_validation.py` | missing file; missing/unexpected column; non-numeric; inf; bad/single-class target; negative Amount/Time; dedup keeps first + counts | M1 |
| `test_split.py` | ratios; stratified prevalence; holdout = latest ~15%; ordering invariant; boundary ties; zero overlap (4 partitions); holdout not stratified | M3 |
| `test_leakage.py` | preprocessor fitted on train only; `Time` not in transformer; CV receives train only; [OPTIONAL] SMOTE affects train only; `train` entrypoint never opens test/holdout paths (monkeypatched reader) | M3/M5 |
| `test_threshold.py` | metric correctness on toy arrays; R_min rule; F2 fallback; tie → higher τ; selection tie-break | M4/M6 |
| `test_inference.py` | `feature_order` 29 without `Time`; shuffled input → identical output; `p ≥ τ` rule; probability ∈ [0,1] | M9 |

### Integration tests (fixture artifact pair; run in CI)

| File | Tests | Milestone |
|---|---|---|
| `test_artifacts.py` | save/load round-trip identical probabilities; startup rejects missing/corrupt model or config, bad `feature_order`, `Time` in `feature_order`, bad τ | M7/M9 |
| `test_api.py` | health; valid predict; response schema; probability range; returned τ; decision rule + label; missing/extra/`Time`/`Class`/string/null/NaN/Inf/negative Amount → 422; key-order independence | M9 |

### Artifact / smoke tests (`@pytest.mark.artifact`; local only)

| Test | Check |
|---|---|
| Real artifact consistency | `model_config.json` τ == `train_metadata.json` τ; all DOC-02 §15 + lineage keys present; no placeholder values |
| Offline/API parity | API probability for a `test.csv` row == `pipeline.predict_proba` |
| Docker/local parity (manual scripted check, M10) | Same request to local uvicorn and container → identical JSON |

## 16. Git Commit / Milestone Strategy

Single `main` branch; optional short-lived feature branches [INHERITED MD-02].

| Milestone | Commit prefix/example | Tag |
|---|---|---|
| M0 | `chore: skeleton, pinned env, DVC raw data` | — |
| M1 | `data: validation and deduplication` | — |
| M2 | `docs: EDA report` | — |
| M3 | `ml: temporal carve-out, split, preprocessing` | `v0.1-data` |
| M4 | `ml: candidate training` | — |
| M5 | `mlops: tuning + MLflow candidate runs` | — |
| M6 | `ml: selection + validation threshold` | — |
| M7 | `ml: test + temporal evaluation` | — |
| M8 | `mlops: dvc pipeline + lock` | `v1.0-model` (after `dvc repro` of record) |
| M9 | `serve: FastAPI service` | — |
| M10 | `docker: serving image` | — |
| M11 | `test: full suite` · `ci: workflow` [OPTIONAL] | — |
| M12 | `docs: README + audit` | `v1.0-serve` |

Always commit `dvc.lock` with the code/config that produced it. `v1.0-model` is placed after M8 so the tag points to the committed `dvc.lock` of record [IMP].

## 17. Two-Day Execution Plan

No durations are estimated per task; blocks are ordered by priority.

**Day 1 — ML core**

| Block | Milestones | Exit condition |
|---|---|---|
| Morning A | M0 setup, M1 validation/dedup | Report matches verified facts |
| Morning B | M2 EDA (keep brief), M3 split + preprocessing | `v0.1-data` tagged |
| Afternoon A | M4 training, M5 tuning + MLflow | 3 candidate runs visible |
| Afternoon B | M6 threshold + selection, M7 primary test evaluation | τ frozen; `test_metrics.json` |

**Day 2 — Temporal, MLOps, serving**

| Block | Milestones | Exit condition |
|---|---|---|
| Morning A | M7 temporal evaluation, M8 DVC pipeline + reproducibility | `v1.0-model` tagged |
| Morning B | M9 FastAPI | DR-02…DR-04 |
| Afternoon A | M10 Docker, M11 tests (+ optional CI) | DR-05…DR-07 |
| Afternoon B | M12 README + §22 audit | Definition of Done |

| Priority | Items |
|---|---|
| **Must-have** | M0–M12 core: validation, dedup, EDA (≤ 8 figs), temporal carve-out, split, preprocessing, 3 models, train-only tuning, MLflow candidate + final runs, validation threshold, test once, **temporal robustness once**, DVC 5-stage pipeline + lock, FastAPI, Docker, unit + integration tests, README |
| **Should-have** | Artifact-marked smoke/parity tests; DVC plots; Docker HEALTHCHECK; non-root user |
| **Bonus [OPTIONAL]** | GitHub Actions CI; SMOTE+LR experiment; bootstrap CIs; EDA notebook; ruff lint |

If time runs short, drop bonus items first, then should-haves. Never cut temporal robustness, leakage tests, or the DVC/MLflow/API/Docker path.

## 18. Acceptance Criteria Matrix

| ID | Requirement / Deliverable | Validation method | Expected result | Milestone |
|---|---|---|---|---|
| AC-01 | FR-001 load dataset | Run `validate` with missing path | Clear error; normal run succeeds | M1 |
| AC-02 | FR-002 schema | `test_validation.py` | Missing col fails; extra col warned | M1 |
| AC-03 | FR-003 missing/inf | Report + tests | 0 missing, 0 inf reported | M1 |
| AC-04 | FR-004 duplicates | `validation_report.json` | 1,081 removed; 283,726 rows | M1 |
| AC-05 | FR-005 target | Tests + report | Only {0,1}, both present | M1 |
| AC-06 | FR-006 class distribution | Reports | Counts + prevalence for raw, clean, each split, holdout | M1/M3 |
| AC-07 | FR-007 stratified split | `split_summary.json`, tests | 70/15/15 of pool; prevalence within tolerance | M3 |
| AC-08 | FR-008 train-only preprocessing | `test_leakage.py` | Pass | M3 |
| AC-09 | FR-009 imbalance | MLflow params | class weights / `scale_pos_weight` logged | M5 |
| AC-10 | FR-010 three models | MLflow | 3 candidate runs | M5 |
| AC-11 | FR-011 comparison | `model_comparison.csv` | All metrics for all candidates | M6 |
| AC-12 | FR-012 metrics | Tests + reports | P, R, F1, F2, PR-AUC, ROC-AUC, CM present | M6/M7 |
| AC-13 | FR-013 threshold | `test_threshold.py`, `train_metadata.json` | Rule-based τ + objective recorded | M6 |
| AC-14 | FR-014 test once | Stage graph + commit history | Single evaluation of record | M7/M8 |
| AC-15 | FR-015 artifacts | Files + round-trip test | `model.joblib` + `model_config.json` valid | M7 |
| AC-16 | FR-016 inference | `test_inference.py` | Probability, class, τ returned | M9 |
| AC-17 | FR-017 temporal robustness | `reports/temporal/*` | All §12 outputs populated | M7 |
| AC-18 | Temporal isolation | `dvc dag`, `test_split.py` | No holdout edge into train/evaluate; zero overlap | M3/M8 |
| AC-19 | NFR-001/002 reproducibility | Fresh clone `dvc repro` | Identical splits; metrics within tolerance | M8/M12 |
| AC-20 | NFR-008 pinned deps | Inspect files | All `==` pins; shared pins identical | M0/M10 |
| AC-21 | NFR-009 no leakage | §10 checklist | All items pass | M3–M8 |
| AC-22 | NFR-010 schema consistency | Tests | 29 features, no `Time`, enforced order | M3/M9 |
| AC-23 | DOC-01 SC-06 beats baseline | `model_comparison.csv` | Selected PR-AUC ≥ LR PR-AUC and > prevalence [EMPIRICAL] | M6 |
| AC-24 | DOC-02 selection framework | `model_selection.md` | Documented rule application | M6 |
| AC-25 | DOC-03 MAC-01…MAC-09 | DOC-03 §14 | All pass | M8/M11 |
| AC-26 | DOC-04 DR-01…DR-08 | DOC-04 §13 | All pass | M9–M12 |
| AC-27 | API contract | `test_api.py` | All cases in §13 pass | M9 |
| AC-28 | Docker boundary | `docker run … ls`, image inspect | No data/MLflow/training code | M10 |
| AC-29 | Documentation | README review | Setup, reproduce, API, Docker, results, limitations | M12 |

## 19. Definition of Done

The project is complete when **all** hold:

1. `dvc repro` from a fresh clone reproduces the pipeline; `dvc status` clean.
2. LR, RF, XGBoost trained, tuned on train only, and logged as MLflow candidate runs.
3. Final MLflow run contains selection, τ, val/test/temporal metrics and artifacts, with lineage tags.
4. τ selected on validation by the documented rule; recorded identically in `train_metadata.json`, `model_config.json` and MLflow.
5. Primary test evaluated once for the release; temporal holdout evaluated once; comparison report written.
6. `models/model.joblib` + `models/model_config.json` exist and pass round-trip and startup checks.
7. FastAPI `/health` and `/predict` behave per DOC-04 locally and in Docker.
8. `pytest` passes (all markers locally; `not artifact` in CI if enabled).
9. README and DOC-01–05 complete; README results match generated reports.
10. No placeholder values in any report, config or README.
11. No known violation of DOC-01–05; any deviation logged in §23.

## 20. Troubleshooting / Failure Handling

| Problem | Symptom | Likely cause | Safe fix | Do NOT |
|---|---|---|---|---|
| DVC stage failure | `dvc repro` errors | Missing dep/out declaration, path typo, two stages writing one file | Fix `dvc.yaml`; run stage module directly to debug | Remove leakage-guard deps restrictions |
| Dataset missing | `validate` hard-fails | `dvc pull` not run / remote unavailable | `dvc pull` or re-download CSV; confirm `dvc status` matches MD5 | Commit CSV to Git |
| Duplicate mismatch | ≠ 1,081 removed | Float parsing changed, subset of columns used, index column read | Read with default pandas parsing, compare all 31 columns | Hard-code the count |
| Too few frauds | Holdout fraud count < 20 | Chronological fraud distribution | Report low-confidence flag | Change `holdout_fraction` after seeing metrics |
| Training too slow | RF search dominates | Large forests × CV | Lower `n_iter` to 5 in config, log it (DOC-02 §10) | Subsample training data; drop a model |
| MLflow not opening | UI empty/errors | Wrong working dir or URI | Run `mlflow ui --backend-store-uri file:./mlruns` from repo root; SQLite fallback | Add a remote tracking server |
| Artifact incompatible | Load error / startup check fails | Library version drift | Reinstall pinned versions; rebuild artifacts with `dvc repro` | Unpin versions or skip the check |
| Version mismatch | DD-07 startup failure in Docker | Serve pins differ from training | Align `requirements-serve.txt` with `requirements.txt` | Relax version check |
| API model loading failure | Service exits at startup | Paths/env vars; `models/` not populated | Set `MODEL_PATH`/`MODEL_CONFIG_PATH`; `dvc pull` | Retrain at startup |
| Docker build failure | Build errors | Missing `models/`, wrong `PYTHONPATH`, wheel issues | `dvc pull` first; check COPY paths; use slim image matching Python version | Copy `data/` or entire `src/` |
| API schema mismatch | Startup "Feature schema incompatible" | `feature_order` ≠ schema fields | Fix schema to match 29-feature config | Add `Time` to schema |
| Artifact tests failing | `-m artifact` failures | Stale artifacts vs code | `dvc repro`; recheck τ consistency | Delete or weaken the tests |

Never change methodology (splits, metrics, threshold rule, models) to make a test pass.

## 21. Scope Protection Rules

Unless DOC-01–05 are formally changed, **DO NOT add**: additional datasets · additional ML models · Streamlit or any frontend · cloud deployment · Kubernetes · MLflow Model Registry · Airflow · Kafka · Terraform · Redis · any database (MLflow's optional SQLite backend excepted, DOC-03) · authentication system · monitoring stack · batch prediction endpoint · feature engineering from `Time`.

A feature being interesting is not a reason to add it. Proposed additions go to §23 as "deferred / out of scope".

## 22. Final Pre-Submission Audit

| Area | Checks |
|---|---|
| ML correctness | [ ] 3 candidates in `model_comparison.csv` · [ ] selection per `model_selection.md` matches DOC-02 §12 · [ ] accuracy not used for selection |
| Leakage | [ ] §10 all ticked · [ ] `pytest tests/test_leakage.py tests/test_split.py` |
| Temporal robustness | [ ] all §12 outputs exist · [ ] interpretation wording correct · [ ] prevalence reported alongside PR-AUC |
| MLOps reproducibility | [ ] fresh clone → `dvc pull && dvc repro` → `dvc status` clean · [ ] `dvc metrics diff` within tolerance |
| MLflow | [ ] `mlflow ui`: 3 (+1) candidate runs, 1 final run · [ ] final run tags match `model_config.json` |
| DVC pipeline | [ ] `dvc dag` has 5 stages · [ ] dependency rules of §8 hold |
| Artifact integrity | [ ] τ identical across `train_metadata.json`, `model_config.json`, MLflow · [ ] `feature_order` 29, no `Time` · [ ] `model_version` = `<semver>+<sha>` |
| API | [ ] `/health` 200 · [ ] `/docs` renders · [ ] valid `/predict` · [ ] `Time` → 422 |
| Docker | [ ] image builds from clean checkout · [ ] parity with local · [ ] contents exclude data/MLflow/training code |
| Tests | [ ] `pytest` green · [ ] [OPTIONAL] CI green |
| Git cleanliness | [ ] `git status` clean · [ ] no CSV/joblib/mlruns in Git (`git ls-files \| grep -E "\.csv$\|\.joblib$\|mlruns"` returns nothing) · [ ] tags `v0.1-data`, `v1.0-model`, `v1.0-serve` |
| Documentation | [ ] README sections complete · [ ] results match reports · [ ] no placeholders (`grep -ri "TODO\|TBD\|<placeholder>"` empty) |

## 23. Decision Log

**Inherited (confirmed upstream, not reopened):** Time ordering-only (D-11/TD-04a); 29 features; dedup pre-split (TD-01); temporal 15% carve-out before 70/15/15 stratified split (TD-17/18); cost-sensitive primary imbalance (TD-06); three models (D-02); train-only RandomizedSearchCV (TD-09); validation-only threshold with R_min rule + F2 fallback (D-04/05); test once, temporal once (D-06/TD-19); artifact pair (D-07); five DVC stages (MD-05); `train_metadata.json` → `model_config.json` (MD-07); MLflow local, no registry (MD-09); FastAPI contract (DD-01…09); Docker boundary (DD-11/12).

**Implementation choices made in this document [IMP]:**

| ID | Choice | Reason | Consistency |
|---|---|---|---|
| IMP-01 | Minimal `pyproject.toml` for `pip install -e .` and pytest marker registration | Makes `python -m fraud_detection.…` work in DVC stages and tests without PATH hacks | Packaging metadata only; no new tool; Docker still uses `PYTHONPATH` per DOC-04 |
| IMP-02 | EDA implemented as `src/fraud_detection/eda.py` (notebook optional) | Scriptable and reproducible | DOC-03 MD-16: EDA outside DVC |
| IMP-03 | Internal row-id column during splitting for overlap tests, dropped before modelling | Needed to prove zero overlap after dedup | Not a feature; never reaches model |
| IMP-04 | `v1.0-model` tag placed after M8 | Tag must point at the `dvc.lock` of record | Refines DOC-03 tag timing |
| IMP-05 | `src/fraud_detection/__init__.py` has no imports | Allows serving image without training modules | Supports DOC-03 import rule |
| IMP-06 | `.gitignore`/`.dockerignore` contents as listed in M0/§14 | Enforces DOC-03/04 boundaries | — |

**To be finalized empirically [EMPIRICAL]:** exact Python version and library pins; Docker base tag; `r_min` confirmation (before M7); selected model; τ; post-dedup fraud count; temporal holdout size and fraud count; whether RF `n_iter` must drop to 5; bootstrap CI inclusion.

**Optional [OPTIONAL]:** GitHub Actions CI; SMOTE+LR; bootstrap CIs; EDA notebook; ruff; Docker HEALTHCHECK.

**Deviation register:** *(record any implementation-time deviation here with ID, description, justification, affected document.)*

| ID | Description | Justification | Affected documents |
|---|---|---|---|
| DEV-01 | Validation report written to `reports/validation/validation_report.json` instead of `reports/validation_report.json`; path set by `paths.validation_report` in `config/config.yaml`. | Requested in the M1 implementation brief; keeps validation outputs grouped like `reports/temporal/` and `reports/eda/`. Configurable, so the pipeline logic is unaffected. `dvc.yaml` (M8) must declare this path. | DOC-02 §1, §17; DOC-03 §4.2; DOC-05 §3, M1 |
| DEV-02 | EDA implemented as package `src/fraud_detection/eda/` (`analysis.py`, `plots.py`, `report.py`, `cli.py`, `__main__.py`) instead of the single module `eda.py` (IMP-02). | Requested in the M2 implementation brief; separates pure statistics (unit-tested) from plotting and report rendering. The command `python -m fraud_detection.eda` is unchanged. The M10 `.dockerignore` entry `src/fraud_detection/eda.py` becomes `src/fraud_detection/eda/`. | DOC-05 §3, M2, §6, §14, IMP-02 |
| DEV-03 | EDA findings written to `reports/eda/eda_report.md` (instead of `eda_summary.md`), figures to `reports/eda/figures/`, plus a machine-readable `reports/eda/eda_summary.json`. Output directory set by `paths.eda_dir`. | Requested in the M2 implementation brief. Still under `reports/eda/`, still ≤ 8 figures, still descriptive only and outside the DVC pipeline. | DOC-02 §4, §17; DOC-05 §3, M2 |
