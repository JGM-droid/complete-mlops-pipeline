# Complete MLOps Pipeline

## Project overview
This repository contains a phase-gated implementation of a tabular machine-learning operations workflow for an employee-attrition prediction use case. The current phase includes deterministic training, MLflow tracking, and controlled experiment comparison.

## Assignment objective
Deliver a reproducible MLOps project that satisfies TripleTen grading requirements for data versioning, configuration management, experiment tracking, testing, CI controls, and drift monitoring.

## Current project status
- Current phase: Phase 6 (deterministic drift monitoring)
- Implemented now: scaffolding, Git/DVC initialization, configuration/validation/preprocessing, deterministic training, MLflow run tracking, five controlled experiments, comparison output, production-style GitHub Actions CI, deterministic drift batch generation, and Evidently-based drift monitoring
- Planned later phases: final submission hardening

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

## GitHub Actions CI
The CI workflow lives at [`.github/workflows/ci.yml`](.github/workflows/ci.yml). It runs on pushes to `main`, pull requests targeting `main`, and manual workflow dispatch.

CI performs the following checks:
- installs the project with the repository's existing dependency convention (`python -m pip install -r requirements.txt` followed by `python -m pip install -e .`)
- prints Python and dependency versions for diagnostics
- runs `python -m compileall src tests`
- runs `python -m pytest tests/ -v`
- verifies the tracked raw CSV and its `.dvc` pointer are present
- runs `python -m dvc status`
- runs `python -m mlops_pipeline.train --config configs/train.yaml`
- generates deterministic drift batches and runs `python -m mlops_pipeline.monitor_drift --config configs/train.yaml --current-batch stable`
- runs the deliberate drift gate path with `python -m mlops_pipeline.monitor_drift --config configs/train.yaml --current-batch drifted` and verifies the nonzero exit status without failing the workflow
- checks repository hygiene with `git status --short` and a non-destructive untracked-file check

Local equivalents:
```powershell
python -m compileall src tests
python -m pytest tests/ -v
python -m mlops_pipeline.train --config configs/train.yaml
python -m dvc status
```

Data and DVC limitation:
- the raw IBM attrition dataset is committed in `data/raw/`
- the repository currently has a `.dvc` pointer for the CSV, but no `dvc.yaml` pipeline file
- because there is no pipeline file and no remote credentials are required for the tracked raw dataset, CI uses `dvc status` and presence checks instead of `dvc repro --dry`

How to read a failure:
- compile or pytest failures point to code regressions
- baseline training failures mean the configured quality gate did not pass
- DVC failures indicate the tracked raw data or DVC metadata is missing or inconsistent
- hygiene failures mean the workflow produced unexpected tracked files or unignored outputs

## Test command
```powershell
# Phase 2 through Phase 4 validation commands
pytest tests/test_config.py tests/test_preprocess.py tests/test_data_validation.py -v
pytest tests/test_evaluation.py tests/test_model_validation.py -v
pytest tests/test_phase4_mlflow.py tests/test_phase4_comparison.py -v
pytest tests/test_phase5_ci.py -v
pytest tests/ -v
```

## Phase 6 drift monitoring
Drift monitoring compares a current batch to a deterministic reference batch using Evidently 0.7.21.

What drift means here:
- drift means the feature distribution in the current batch moved far enough from the reference batch to exceed the configured dataset-drift share threshold
- drift does not automatically mean model failure; it is a signal to review data quality, feature stability, and downstream model impact

How the batches are built:
- the reference batch is a deterministic sample from the canonical IBM HR CSV
- the stable current batch is a deterministic row-order permutation of that reference batch
- the drifted current batch starts from the stable batch and mutates only configured feature columns, never `Attrition`

Commands:
```powershell
python -m mlops_pipeline.monitor_drift --config configs/train.yaml --current-batch stable
python -m mlops_pipeline.monitor_drift --config configs/train.yaml --current-batch drifted
```

Outputs:
- `reports/drift_reference.csv`
- `reports/drift_stable.csv`
- `reports/drift_drifted.csv`
- `reports/drift_summary.json`
- `reports/drift_report.html`

Threshold and rationale:
- `monitoring.dataset_drift_threshold` is set to `0.15`
- `monitoring.feature_drift_threshold` is set to `0.5`
- the configured numeric and categorical shifts are intentionally strong enough to make the drifted batch fail while the stable batch remains below the gate

Demonstration result:
- stable batch: exits `0`, `gate_status = passed`
- drifted batch: exits `1`, `gate_status = failed`

## CI/CD status
GitHub Actions CI is implemented in [`.github/workflows/ci.yml`](.github/workflows/ci.yml). It validates the repository on pushes to `main`, pull requests to `main`, and manual dispatch.

The workflow blocks merge readiness on compile, test, DVC, baseline training, drift-monitoring gate, and hygiene failures. GitHub-hosted execution still needs a pushed run for remote evidence, so the workflow is locally validated but not claimed green on GitHub yet.

## Project roadmap
See `docs/roadmap.md` for phase-by-phase deliverables, acceptance checks, and statuses.

## Grader workflow
1. Create and activate Python 3.11 environment.
2. Install dependencies from `requirements.txt`.
3. Review governing documentation in `docs/`.
4. Ensure `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` is present.
5. Run dataset audit command and inspect JSON report.
6. Run the Phase 4 experiment command and compare the completed runs.
7. Run the CI checks locally with `python -m compileall src tests`, `python -m pytest tests/ -v`, `python -m mlops_pipeline.train --config configs/train.yaml`, and `python -m dvc status`.
8. Continue with later phases once the current phase acceptance checks are approved.

## Known Phase 6 limitations
- GitHub-hosted CI evidence is still pending because the workflow has not been pushed and observed in Actions yet.
- Drift monitoring is intentionally conservative and only validates the configured batch comparison path; it does not trigger retraining automatically.
- Experiment artifacts are recorded locally in `mlruns/` and are intentionally not committed.
