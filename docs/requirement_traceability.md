# Requirement Traceability Matrix

| Requirement | Planned implementation location | Required evidence | Current status | Grading risk |
|---|---|---|---|---|
| Git and repository structure | Repository root, `src/`, `tests/`, `docs/`, `configs/`, `data/` | Directory tree, `git status` | In progress | Low |
| DVC | `.dvc/`, `.dvcignore`, tracked data metadata files, `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv.dvc` | `dvc --version`, `dvc status`, DVC metadata, Git-visible `.dvc` pointer | In progress | Medium |
| YAML configuration | `configs/train.yaml`, `src/mlops_pipeline/config.py` | YAML files with confirmed dataset path, target, exclusion list, deterministic missingness settings, and passing config tests | Complete | Low |
| MLflow parameters | `src/mlops_pipeline/experiment_runner.py`, `src/mlops_pipeline/phase4_support.py` | MLflow run parameter logs | Complete | Low |
| MLflow data version | `src/mlops_pipeline/phase4_support.py` | Logged Git commit tag and safe DVC-status metadata in MLflow run | Complete | Low |
| MLflow metrics | `src/mlops_pipeline/evaluate.py`, `src/mlops_pipeline/phase4_support.py` | Local evaluation metrics computed and logged to MLflow | Complete | Low |
| MLflow model artifact | `src/mlops_pipeline/train.py`, `src/mlops_pipeline/phase4_support.py` | Registered sklearn Pipeline artifact in MLflow | Complete | Low |
| Five experiment runs | `configs/experiments/`, `src/mlops_pipeline/experiment_runner.py` | Five committed configs and five completed local MLflow run records | Complete | Low |
| `mlflow.search_runs` comparison | `src/mlops_pipeline/compare_experiments.py`, `src/mlops_pipeline/phase4_support.py` | Comparison output/report and deterministic best-run selection | Complete | Low |
| Six preprocessing unit tests | `tests/test_preprocess.py` | Passing pytest output with substantive preprocessing behaviors | Complete | Low |
| Three dataset-validation tests | `tests/test_data_validation.py` | Passing pytest output on real dataset plus failure-path checks | Complete | Low |
| Two model-validation tests | `tests/test_model_validation.py` | Passing pytest output for prediction-shape/value checks and threshold-quality checks | Complete | Low |
| pytest root command | Project root + `pyproject.toml` | `pytest tests/ -v` execution from root | Complete | Low |
| GitHub Actions triggers | `.github/workflows/ci.yml` | Workflow YAML plus local validation output | Implemented locally; GitHub run evidence pending | Medium |
| Test and training jobs | `.github/workflows/ci.yml` | Separate jobs with `needs: tests` in workflow YAML | Implemented locally; GitHub run evidence pending | Medium |
| Training dependency on tests | `.github/workflows/ci.yml` | Job dependency visible in workflow YAML | Implemented locally; GitHub run evidence pending | Medium |
| Performance quality gate | `src/mlops_pipeline/evaluate.py`, `src/mlops_pipeline/train.py`, `configs/train.yaml` | Failing gate raises error and returns non-zero CLI exit code | Complete | Low |
| Green Actions run | Future CI execution | Successful workflow run screenshot/log | Not started | Medium |
| Evidently feature drift | `src/mlops_pipeline/drift_batches.py`, `src/mlops_pipeline/monitor_drift.py` | Deterministic drift batches plus Evidently comparison output | Complete locally | Low |
| Drift summary | `reports/drift_summary.json` | Drift summary JSON with reference/current paths, drift counts, drift percentage, drift names, threshold, gate status, and Evidently version | Complete locally | Low |
| HTML report | `reports/drift_report.html` | Drift HTML report | Complete locally | Low |
| Configurable drift threshold | `configs/train.yaml`, `src/mlops_pipeline/config.py` | `monitoring.dataset_drift_threshold` and `monitoring.feature_drift_threshold` are validated and exercised in tests | Complete locally | Low |
| Exit code 1 | `src/mlops_pipeline/monitor_drift.py` | Terminal exit code on threshold breach | Complete locally | Low |
| Written monitoring analysis | `README.md`, `docs/project_walkthrough.md`, `docs/decision_log.md` | Phase 6 analysis and interview talking points | Complete locally | Low |
| Deterministic monitoring batches | `src/mlops_pipeline/drift_batches.py` | Stable and drifted batches generated from the canonical raw CSV with identical seed-dependent output | Complete locally | Low |
| Monitoring CLI | `src/mlops_pipeline/monitor_drift.py` | `python -m mlops_pipeline.monitor_drift --config configs/train.yaml --current-batch stable|drifted` | Complete locally | Low |
| README setup and execution instructions | `README.md` | Reproducible setup plus Phase 4/5 training, experiment, compare, CI, and UI commands | Complete | Low |

## Phase 5 implementation evidence update

- CI workflow implementation: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on push, pull request, and manual dispatch.
- Validation stages: dependency install, version reporting, compileall, pytest, raw-data presence checks, `dvc status`, baseline training, and hygiene checks.
- DVC choice: `dvc status` is the strongest safe non-destructive check for this repository because the tracked raw dataset is committed locally and there is no `dvc.yaml` pipeline file to dry-run.
- MLflow isolation: CI and tests use `MLOPS_PIPELINE_MLFLOW_TRACKING_URI` so run data is written outside the repository checkout.
- Failure behavior: any compile, test, DVC, baseline training, or hygiene failure blocks the pipeline.

## Phase 6 implementation evidence update

- Deterministic batch generation: `src/mlops_pipeline/drift_batches.py` builds a reference batch, a stable row-order permutation, and a deliberately drifted current batch without mutating `Attrition` or the raw CSV.
- Drift monitoring implementation: `src/mlops_pipeline/monitor_drift.py` uses Evidently 0.7.21 with `Report`, `DataDriftPreset`, `DataDefinition`, and `Dataset.from_pandas` to compare current versus reference batches.
- Output artifacts: `reports/drift_summary.json` and `reports/drift_report.html` are generated deterministically along with the three CSV batch files.
- Configurable gate: `configs/train.yaml` defines the random seed, batch sizes, drift threshold, drifted feature names, per-feature drift settings, and output paths.
- Test evidence:
	- `pytest tests/test_drift_monitoring.py -v` passed.
	- `pytest tests/ -v` passed after the Phase 6 changes.

## Dataset audit evidence update

- Selected dataset: IBM HR Analytics Employee Attrition
- Source file: `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Target column: `Attrition`
- Rows: 1,470
- Columns: 35 total (34 excluding target)
- Target classes: `No` (1,233, 83.8776%) and `Yes` (237, 16.1224%)
- Duplicate rows: 0
- Missing values: 0
- Evidence artifact: `reports/dataset_audit.json` (local generated report, ignored by Git)
- Requirement implication: deterministic missingness simulation is required in a later phase to satisfy assignment expectations around missing-data handling.

## Phase 2 implementation evidence update

- Configuration implementation: `src/mlops_pipeline/config.py` with strict section/type/path validation and project-owned error handling.
- Dataset validation implementation: `src/mlops_pipeline/data_validation.py` with schema, target, uniqueness, constant-column, duplicates, row-count, feature-count, and range checks.
- Preprocessing implementation: `src/mlops_pipeline/preprocess.py` with feature/target separation, exclusion handling, deterministic feature-only missingness simulation, and sklearn `ColumnTransformer` assembly.
- Missingness policy evidence: configured in `configs/train.yaml` under `features.missingness_simulation` with `enabled: true`, `fraction: 0.04`, `random_seed: 42`, and `strategy: all_eligible`.
- Test evidence:
	- `pytest tests/test_config.py tests/test_preprocess.py tests/test_data_validation.py -v` passed.
	- `pytest tests/ -v` passed.

## Phase 3 implementation evidence update

- Training orchestration: `src/mlops_pipeline/train.py` implements deterministic split, leakage-safe preprocessing fit, Logistic Regression baseline training, metric evaluation, and quality-gate enforcement.
- Evaluation: `src/mlops_pipeline/evaluate.py` computes accuracy, balanced accuracy, precision/recall/F1 for Attrition=Yes, ROC AUC, and confusion matrix.
- Baseline settings: `model.algorithm=logistic_regression`, `class_weight=balanced`, `max_iter=3000`.
- Primary metric: `f1_attrition` with threshold `0.45` configured in `configs/train.yaml`.
- Baseline run evidence (deterministic repeated runs):
	- `f1_attrition`: 0.4962
	- `balanced_accuracy`: 0.7438
	- `roc_auc`: 0.8103
	- `precision_attrition`: 0.3837
	- `recall_attrition`: 0.7021
	- `accuracy`: 0.7721
- Test evidence:
	- `pytest tests/test_evaluation.py tests/test_model_validation.py -v` passed.
	- `pytest tests/ -v` passed.

## Phase 4 implementation evidence update

- MLflow tracking implementation: `src/mlops_pipeline/phase4_support.py` logs parameters, metrics, artifacts, tags, and the sklearn Pipeline model to the local `mlruns/` store.
- Experiment execution implementation: `src/mlops_pipeline/experiment_runner.py` runs the five controlled experiment YAML files sequentially and writes a machine-readable summary.
- Comparison implementation: `src/mlops_pipeline/compare_experiments.py` ranks completed runs by `f1_attrition` first and excludes failed quality-gate runs from best-run selection.
- Experiment configurations:
	- `configs/experiments/01_logreg_balanced.yaml`
	- `configs/experiments/02_logreg_stronger_regularization.yaml`
	- `configs/experiments/03_logreg_no_class_weight.yaml`
	- `configs/experiments/04_random_forest_balanced.yaml`
	- `configs/experiments/05_gradient_boosting.yaml`
- Completed run metrics:
	- `exp-01-logreg-balanced`: F1 0.4962, Recall 0.7021, Precision 0.3837, Balanced Accuracy 0.7438, ROC AUC 0.8103, Accuracy 0.7721, Quality Gate passed
	- `exp-02-logreg-stronger-regularization`: F1 0.5075, Recall 0.7234, Precision 0.3908, Balanced Accuracy 0.7544, ROC AUC 0.8083, Accuracy 0.7755, Quality Gate passed
	- `exp-03-logreg-no-class_weight`: F1 0.4928, Recall 0.3617, Precision 0.7727, Balanced Accuracy 0.6707, ROC AUC 0.8141, Accuracy 0.8810, Quality Gate passed
	- `exp-04-random-forest-balanced`: F1 0.4615, Recall 0.5106, Precision 0.4211, Balanced Accuracy 0.6885, ROC AUC 0.7735, Accuracy 0.8095, Quality Gate passed
	- `exp-05-gradient-boosting`: F1 0.2667, Recall 0.1702, Precision 0.6154, Balanced Accuracy 0.5750, ROC AUC 0.7696, Accuracy 0.8503, Quality Gate failed
- Best eligible run: `exp-02-logreg-stronger-regularization` with `f1_attrition=0.5075`.
- Selection rule: highest `f1_attrition`, then `balanced_accuracy`, `roc_auc`, `precision_attrition`, `recall_attrition`, `accuracy`, run name, and run ID; failed-gate runs are excluded from selection.
- Test evidence:
	- `pytest tests/test_phase4_mlflow.py tests/test_phase4_comparison.py -v` passed.
	- `pytest tests/ -v` passed.
