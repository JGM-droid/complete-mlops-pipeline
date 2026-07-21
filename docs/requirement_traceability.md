# Requirement Traceability Matrix

| Requirement | Planned implementation location | Required evidence | Current status | Grading risk |
|---|---|---|---|---|
| Git and repository structure | Repository root, `src/`, `tests/`, `docs/`, `configs/`, `data/` | Directory tree, `git status` | In progress | Low |
| DVC | `.dvc/`, `.dvcignore`, tracked data metadata files, `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv.dvc` | `dvc --version`, `dvc status`, DVC metadata, Git-visible `.dvc` pointer | In progress | Medium |
| YAML configuration | `configs/train.yaml`, future config loaders in `src/mlops_pipeline/config.py` | YAML files with confirmed dataset path, target, and exclusion list; config-loading tests later | In progress | Medium |
| MLflow parameters | `src/mlops_pipeline/experiment_runner.py` (future) | MLflow run parameter logs | Not started | Medium |
| MLflow data version | Future training and experiment modules | Logged DVC hash/tag in MLflow run | Not started | High |
| MLflow metrics | `src/mlops_pipeline/evaluate.py` + experiment logging (future) | MLflow metric history | Not started | Medium |
| MLflow model artifact | `src/mlops_pipeline/train.py` + MLflow logging (future) | Registered or logged model artifact | Not started | Medium |
| Five experiment runs | `configs/experiments/` + runner module (future) | Five committed configs and five run records | Not started | High |
| `mlflow.search_runs` comparison | `src/mlops_pipeline/compare_experiments.py` (future) | Comparison output/report | Not started | Medium |
| Six preprocessing unit tests | `tests/test_preprocess.py` (future) | Passing pytest output | Not started | High |
| Three dataset-validation tests | `tests/test_data_validation.py` (future) | Passing pytest output including deterministic missingness simulation checks | Not started | High |
| Two model-validation tests | `tests/test_model_validation.py` (future) | Passing pytest output | Not started | High |
| pytest root command | Project root + `pyproject.toml` | `pytest` execution from root | Not started | Medium |
| GitHub Actions triggers | Future workflow files in `.github/workflows/` | Workflow YAML and run history | Not started | Medium |
| Test and training jobs | Future CI workflows | Separate jobs in CI logs | Not started | Medium |
| Training dependency on tests | Future CI workflow graph | Job dependency visible in Actions graph | Not started | Medium |
| Performance quality gate | Future evaluation and CI job logic | Failing gate blocks promotion | Not started | High |
| Green Actions run | Future CI execution | Successful workflow run screenshot/log | Not started | Medium |
| Evidently feature drift | `src/mlops_pipeline/monitor_drift.py` (future) | Drift detection output | Not started | Medium |
| Drift summary | `reports/` outputs from monitoring (future) | Drift summary JSON | Not started | Medium |
| HTML report | `reports/` outputs from monitoring (future) | Drift HTML report | Not started | Medium |
| Configurable drift threshold | `configs/train.yaml` + monitoring module (future) | Threshold in config and test evidence | In progress | Medium |
| Exit code 1 | `src/mlops_pipeline/monitor_drift.py` (future) | Terminal exit code on threshold breach | Not started | Medium |
| Written monitoring analysis | Documentation and reports (future) | Analysis section in report/readme | Not started | Medium |
| README setup and execution instructions | `README.md` | Reproducible step-by-step setup and command docs with selected dataset details | In progress | Low |

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
