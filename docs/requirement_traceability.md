# Requirement Traceability Matrix

| Requirement | Planned implementation location | Required evidence | Current status | Grading risk |
|---|---|---|---|---|
| Git and repository structure | Repository root, `src/`, `tests/`, `docs/`, `configs/`, `data/` | Directory tree, `git status` | In progress | Low |
| DVC | `.dvc/`, `.dvcignore`, tracked data metadata files, `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv.dvc` | `dvc --version`, `dvc status`, DVC metadata, Git-visible `.dvc` pointer | In progress | Medium |
| YAML configuration | `configs/train.yaml`, `src/mlops_pipeline/config.py` | YAML files with confirmed dataset path, target, exclusion list, deterministic missingness settings, and passing config tests | Complete | Low |
| MLflow parameters | `src/mlops_pipeline/experiment_runner.py` (future) | MLflow run parameter logs | Not started | Medium |
| MLflow data version | Future training and experiment modules | Logged DVC hash/tag in MLflow run | Not started | High |
| MLflow metrics | `src/mlops_pipeline/evaluate.py` + experiment logging (future) | Local evaluation metrics computed; MLflow logging deferred | In progress | Medium |
| MLflow model artifact | `src/mlops_pipeline/train.py` + MLflow logging (future) | Registered or logged model artifact | Not started | Medium |
| Five experiment runs | `configs/experiments/` + runner module (future) | Five committed configs and five run records | Not started | High |
| `mlflow.search_runs` comparison | `src/mlops_pipeline/compare_experiments.py` (future) | Comparison output/report | Not started | Medium |
| Six preprocessing unit tests | `tests/test_preprocess.py` | Passing pytest output with substantive preprocessing behaviors | Complete | Low |
| Three dataset-validation tests | `tests/test_data_validation.py` | Passing pytest output on real dataset plus failure-path checks | Complete | Low |
| Two model-validation tests | `tests/test_model_validation.py` | Passing pytest output for prediction-shape/value checks and threshold-quality checks | Complete | Low |
| pytest root command | Project root + `pyproject.toml` | `pytest tests/ -v` execution from root | Complete | Low |
| GitHub Actions triggers | Future workflow files in `.github/workflows/` | Workflow YAML and run history | Not started | Medium |
| Test and training jobs | Future CI workflows | Separate jobs in CI logs | Not started | Medium |
| Training dependency on tests | Future CI workflow graph | Job dependency visible in Actions graph | Not started | Medium |
| Performance quality gate | `src/mlops_pipeline/evaluate.py`, `src/mlops_pipeline/train.py`, `configs/train.yaml` | Failing gate raises error and returns non-zero CLI exit code | Complete | Low |
| Green Actions run | Future CI execution | Successful workflow run screenshot/log | Not started | Medium |
| Evidently feature drift | `src/mlops_pipeline/monitor_drift.py` (future) | Drift detection output | Not started | Medium |
| Drift summary | `reports/` outputs from monitoring (future) | Drift summary JSON | Not started | Medium |
| HTML report | `reports/` outputs from monitoring (future) | Drift HTML report | Not started | Medium |
| Configurable drift threshold | `configs/train.yaml` + monitoring module (future) | Threshold in config and test evidence | In progress | Medium |
| Exit code 1 | `src/mlops_pipeline/monitor_drift.py` (future) | Terminal exit code on threshold breach | Not started | Medium |
| Written monitoring analysis | Documentation and reports (future) | Analysis section in report/readme | Not started | Medium |
| README setup and execution instructions | `README.md` | Reproducible setup plus training and test commands for implemented phases | In progress | Low |

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
