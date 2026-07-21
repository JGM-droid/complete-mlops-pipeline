# Complete MLOps Pipeline

## Project overview
This repository contains a phase-gated implementation of a tabular machine-learning operations workflow for an employee-attrition prediction use case. The current phase establishes the repository foundation, dependency baseline, and dataset-audit tooling.

## Assignment objective
Deliver a reproducible MLOps project that satisfies TripleTen grading requirements for data versioning, configuration management, experiment tracking, testing, CI controls, and drift monitoring.

## Current project status
- Current phase: Phase 1 (Repository foundation and dataset audit)
- Implemented now: scaffolding, Git/DVC initialization, dependency verification, governance docs, config skeleton, dataset audit CLI
- Planned later phases: preprocessing, training, evaluation, MLflow experiment operations, CI, and drift monitoring

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
- Assignment implication: the source dataset is fully clean, so deterministic missingness simulation will be introduced in a later phase.

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
# Planned for a later phase; not implemented in Phase 1
python -m mlops_pipeline.train --config configs/train.yaml
```

## Planned test command
```powershell
# Planned for a later phase; placeholder tests only in Phase 1
pytest
```

## Planned MLflow command
```powershell
# Planned for a later phase; experiment runner not implemented in Phase 1
python -m mlops_pipeline.experiment_runner --config configs/experiments/<experiment>.yaml
```

## Planned drift-monitoring command
```powershell
# Planned for a later phase; drift monitoring not implemented in Phase 1
python -m mlops_pipeline.monitor_drift --config configs/train.yaml
```

## CI/CD status
GitHub Actions workflows are planned for later phases. No CI workflow files are implemented in Phase 1.

## Project roadmap
See `docs/roadmap.md` for phase-by-phase deliverables, acceptance checks, and statuses.

## Grader workflow
1. Create and activate Python 3.11 environment.
2. Install dependencies from `requirements.txt`.
3. Review governing documentation in `docs/`.
4. Ensure `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` is present.
5. Run dataset audit command and inspect JSON report.
6. Continue with later phases once Phase 1 is approved.

## Known Phase 1 limitations
- No preprocessing, training, evaluation, or model validation logic yet.
- No MLflow run execution workflow yet.
- No drift monitoring implementation yet.
- No GitHub Actions workflow yet.
- Test modules are placeholders only.
