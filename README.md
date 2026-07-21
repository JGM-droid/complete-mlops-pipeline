# Complete MLOps Pipeline

## Project overview
This repository contains a phase-gated implementation of a tabular machine-learning operations workflow for an employee-attrition prediction use case. The current phase includes deterministic training, MLflow tracking, and controlled experiment comparison.

## Assignment objective
Deliver a reproducible MLOps project that satisfies TripleTen grading requirements for data versioning, configuration management, experiment tracking, testing, CI controls, and drift monitoring.

## Current project status
- Current phase: Phase 4 (MLflow experiment tracking and controlled model comparison)
- Implemented now: scaffolding, Git/DVC initialization, configuration/validation/preprocessing, deterministic training, MLflow run tracking, five controlled experiments, and comparison output
- Planned later phases: drift monitoring and CI

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

## Phase 4 training command
```powershell
python -m mlops_pipeline.train --config configs/train.yaml
```

## Phase 4 experiment command
```powershell
python -m mlops_pipeline.experiment_runner --config-dir configs/experiments
```

## Compare runs command
```powershell
python -m mlops_pipeline.compare_experiments --config configs/train.yaml
```

## Local MLflow UI
```powershell
python -m mlflow ui --backend-store-uri mlruns
```

## Local MLflow storage
Local MLflow data is stored in the repository-root `mlruns/` directory. The directory is ignored by Git so runtime tracking data is kept out of commits.

## Test command
```powershell
# Phase 2 through Phase 4 validation commands
pytest tests/test_config.py tests/test_preprocess.py tests/test_data_validation.py -v
pytest tests/test_evaluation.py tests/test_model_validation.py -v
pytest tests/test_phase4_mlflow.py tests/test_phase4_comparison.py -v
pytest tests/ -v
```

## Planned drift-monitoring command
```powershell
# Planned for a later phase; drift monitoring is not implemented in Phase 4
python -m mlops_pipeline.monitor_drift --config configs/train.yaml
```

## CI/CD status
GitHub Actions workflows are planned for later phases. No CI workflow files are implemented in Phase 4.

## Project roadmap
See `docs/roadmap.md` for phase-by-phase deliverables, acceptance checks, and statuses.

## Grader workflow
1. Create and activate Python 3.11 environment.
2. Install dependencies from `requirements.txt`.
3. Review governing documentation in `docs/`.
4. Ensure `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` is present.
5. Run dataset audit command and inspect JSON report.
6. Run the Phase 4 experiment command and compare the completed runs.
7. Continue with later phases once the current phase acceptance checks are approved.

## Known Phase 4 limitations
- No drift monitoring implementation yet.
- No GitHub Actions workflow yet.
- Experiment artifacts are recorded locally in `mlruns/` and are intentionally not committed.
