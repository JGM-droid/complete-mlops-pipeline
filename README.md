# Complete MLOps Pipeline

## Project overview
This repository contains a phase-gated implementation of a tabular machine-learning operations workflow for an employee-attrition prediction use case. The current phase includes deterministic baseline training and evaluation.

## Assignment objective
Deliver a reproducible MLOps project that satisfies TripleTen grading requirements for data versioning, configuration management, experiment tracking, testing, CI controls, and drift monitoring.

## Current project status
- Current phase: Phase 3 (Training and evaluation)
- Implemented now: scaffolding, Git/DVC initialization, configuration/validation/preprocessing, deterministic Logistic Regression baseline training, evaluation metrics, and quality-gate enforcement
- Planned later phases: MLflow experiment operations, CI, and drift monitoring

## Architecture summary
The project follows an architecture-first, phase-gated engineering process. Implementation changes must comply with governing documentation in `docs/` and pass architecture review and implementation review before phase promotion.

## Repository structure
- `configs/`: training configuration skeleton and future experiment configs
- `data/`: raw, processed, and production datasets
- `docs/`: architecture, roadmap, traceability matrix, and ADR log
- `reports/`: generated audit and monitoring outputs
- `src/mlops_pipeline/`: project modules
- `tests/`: planned unit and validation tests

## Prerequisites
- Windows PowerShell (or any shell with equivalent commands)
- Python 3.11
- Git

## Environment setup
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Dependency installation
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset status
The selected dataset is IBM HR Analytics Employee Attrition.

- Source filename: `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Target column: `Attrition`
- Audit summary: 1,470 rows, 35 total columns (34 excluding target), mixed numeric/categorical data, 0 duplicate rows, and 0 missing values.
- Source-data policy: the raw CSV is immutable and is never modified.
- Assignment implication: because source data is fully clean, deterministic feature-only missingness simulation is applied during preprocessing (never on `Attrition`).

## Planned DVC workflow
```powershell
# Already initialized in Phase 1:
dvc init

# Dataset tracking in Phase 1:
dvc add data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv

# Planned later for standard workflow progression:
git add data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv.dvc .gitignore
```

## Dataset audit command
```powershell
python -m mlops_pipeline.dataset_audit --csv-path data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv --target-column Attrition --save-json --output-json reports/dataset_audit.json
```

## Planned training command
```powershell
python -m mlops_pipeline.train --config configs/train.yaml
```

## Planned test command
```powershell
# Phase 2 and Phase 3 validation commands
pytest tests/test_config.py tests/test_preprocess.py tests/test_data_validation.py -v
pytest tests/test_evaluation.py tests/test_model_validation.py -v
pytest tests/ -v
```

## Planned MLflow command
```powershell
# Planned for Phase 4; experiment runner remains deferred
python -m mlops_pipeline.experiment_runner --config configs/experiments/<experiment>.yaml
```

## Planned drift-monitoring command
```powershell
# Planned for a later phase; drift monitoring not implemented in Phase 3
python -m mlops_pipeline.monitor_drift --config configs/train.yaml
```

## CI/CD status
GitHub Actions workflows are planned for later phases. No CI workflow files are implemented in Phase 3.

## Project roadmap
See `docs/roadmap.md` for phase-by-phase deliverables, acceptance checks, and statuses.

## Grader workflow
1. Create and activate Python 3.11 environment.
2. Install dependencies from `requirements.txt`.
3. Review governing documentation in `docs/`.
4. Ensure `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` is present.
5. Run dataset audit command and inspect JSON report.
6. Continue with later phases once the current phase acceptance checks are approved.

## Known Phase 3 limitations
- No MLflow run execution workflow yet.
- No drift monitoring implementation yet.
- No GitHub Actions workflow yet.

## Phase 3 baseline model decisions
- Baseline algorithm: Logistic Regression
- Class-weighting: `balanced` to mitigate class imbalance (`Attrition=Yes` ~16.1%)
- Primary metric: `f1_attrition` to balance precision/recall for the minority positive class
- Secondary metrics: balanced accuracy, ROC AUC, precision/recall for Attrition=Yes, and accuracy
- Configured primary threshold: `f1_attrition >= 0.45`

## Current observed baseline metrics
- f1_attrition: 0.4962
- balanced_accuracy: 0.7438
- roc_auc: 0.8103
- precision_attrition: 0.3837
- recall_attrition: 0.7021
- accuracy: 0.7721

These values were stable across repeated deterministic runs with the same configuration.

## Artifact ownership note
MLflow artifact logging remains intentionally deferred to Phase 4 to avoid duplicate artifact ownership paths in earlier phases.
