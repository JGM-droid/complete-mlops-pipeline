# Complete MLOps Pipeline

## Problem

The project uses employee-attrition prediction as the ML problem, but the main focus is the workflow around it. Training a model is straightforward; keeping the data, preprocessing, experiments, tests, and drift checks reproducible is the part this repository is meant to show.

## Solution

I built a configuration-driven pipeline around the IBM HR Analytics Employee Attrition dataset. It covers dataset audit, validation, preprocessing, deterministic training, experiment tracking, CI checks, and drift monitoring using Python 3.11, pandas, scikit-learn, PyYAML, DVC, MLflow, pytest, GitHub Actions, and Evidently.

## Engineering highlights

- Deterministic preprocessing and train/test splitting to keep repeated runs comparable.
- Local MLflow tracking with controlled experiment execution across five model configurations.
- Deterministic experiment comparison that selects the best eligible run from completed experiments.
- CI workflow with separate validation, training, and drift-monitoring jobs.
- Drift monitoring that compares stable and intentionally drifted batches and exits nonzero when the configured gate fails.

## Results / Current Status

- Implemented: dataset audit, configuration validation, preprocessing, deterministic training and evaluation, experiment tracking, experiment comparison, CI workflow, and drift monitoring.
- Current best eligible experiment is selected from five completed runs and tracked in MLflow artifacts.
- Repository includes local reports and documentation for architecture, roadmap, and walkthrough details.
- Not included intentionally: serving API, cloud deployment, orchestration, or automatic retraining.

## Repository Structure

- `configs/`: training and experiment configuration files
- `data/`: raw, processed, and production datasets
- `docs/`: architecture, roadmap, walkthrough, and decision records
- `reports/`: dataset audit, drift outputs, and experiment-comparison outputs
- `src/mlops_pipeline/`: application modules
- `tests/`: unit and integration-style validation tests

## Setup and Usage

### Prerequisites

- Windows PowerShell (or any shell with equivalent commands)
- Python 3.11
- Git

### Environment setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Dependency installation

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Dataset

The project uses the IBM HR Analytics Employee Attrition dataset.

- Source filename: `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Target column: `Attrition`
- Audit summary: 1,470 rows, 35 total columns, mixed numeric and categorical features, 0 duplicate rows, and 0 missing values in the source CSV
- Raw-data policy: the source CSV is treated as immutable and is never edited in place
- Preprocessing implication: because the source data is clean, deterministic feature-only missingness simulation is applied during preprocessing and never to `Attrition`

### Common commands

### Dataset audit

```powershell
python -m mlops_pipeline.dataset_audit --csv-path data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv --target-column Attrition --save-json --output-json reports/dataset_audit.json
```

### Baseline training

```powershell
python -m mlops_pipeline.train --config configs/train.yaml
```

### Run controlled experiments

```powershell
python -m mlops_pipeline.experiment_runner --config-dir configs/experiments
```

### Compare experiment runs

```powershell
python -m mlops_pipeline.compare_experiments --config configs/train.yaml
```

### Start the local MLflow UI

```powershell
python -m mlflow ui --backend-store-uri mlruns
```

### Run drift monitoring

```powershell
python -m mlops_pipeline.monitor_drift --config configs/train.yaml --current-batch stable
python -m mlops_pipeline.monitor_drift --config configs/train.yaml --current-batch drifted
```

### DVC commands

The repository already includes DVC tracking for the raw dataset. These commands document the dataset versioning flow used in the project:

```powershell
dvc init
dvc add data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
git add data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv.dvc .gitignore
```

Useful local checks:

```powershell
python -m dvc status
```

The GitHub Actions workflow also runs `python -m dvc pull` when repository secrets for the configured remote are available.

### MLflow storage

Local MLflow tracking data is stored in the repository-root `mlruns/` directory. Runtime tracking data is kept out of normal source changes.

### Additional documentation

- `docs/project_walkthrough.md`: end-to-end implementation details and experiment summary
- `docs/architecture.md`: architecture principles and repository constraints
- `docs/roadmap.md`: phased delivery history and remaining documentation work
- `docs/decision_log.md`: notable design decisions and rationale
- `docs/drift_analysis.md`: concise interpretation of Phase 6 stable and drifted monitoring outcomes

## Testing

### Run tests

```powershell
pytest tests/test_config.py tests/test_preprocess.py tests/test_data_validation.py -v
pytest tests/test_evaluation.py tests/test_model_validation.py -v
pytest tests/test_phase4_mlflow.py tests/test_phase4_comparison.py -v
pytest tests/test_phase5_ci.py -v
pytest tests/ -v
```

### CI validation

GitHub Actions is defined in `.github/workflows/ci.yml` and runs on pushes to `main`, pull requests targeting `main`, and manual workflow dispatch.

Current CI evidence: GitHub Actions run `#29960096801` completed successfully, including test and DVC validation, baseline training and quality-gate enforcement, and drift-monitoring gate validation (stable batch pass plus drifted batch expected gate-failure behavior).

The workflow validates:

- dependency installation and environment setup
- DVC restore and DVC state checks
- source compilation and full pytest execution
- baseline training and quality-gate enforcement
- stable drift pass behavior and drifted-batch gate failure behavior
- repository hygiene after automation steps

Local equivalents:

```powershell
python -m compileall src tests
python -m pytest tests/ -v
python -m mlops_pipeline.train --config configs/train.yaml
python -m dvc status
```

### Drift-monitoring outputs

Running the monitoring commands produces:

- `reports/drift_reference.csv`
- `reports/drift_stable.csv`
- `reports/drift_drifted.csv`
- `reports/drift_summary.json`
- `reports/drift_report.html`

The stable batch is designed to pass the configured gate. The drifted batch is designed to fail it, which makes the monitoring path testable and deterministic.

Interpretation and follow-up guidance are documented in `docs/drift_analysis.md`.

## Limitations

- This repository demonstrates a local and CI-validated pipeline, not a deployed inference service.
- Drift monitoring raises a deterministic signal but does not trigger retraining automatically.
- Experiment artifacts are stored locally in `mlruns/` and are not intended to be committed as normal source history.
