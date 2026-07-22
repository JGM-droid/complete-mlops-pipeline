# Decision Log (ADR Style)

## ADR-001: Adopt `src/mlops_pipeline` package layout
- Status: Accepted
- Context: The project needs a scalable package layout that supports clean imports and modular development.
- Decision: Use a standard `src` layout with package root at `src/mlops_pipeline/`.
- Consequences: Import behavior is explicit and packaging/tooling is simpler to standardize.

## ADR-002: Select IBM HR Analytics Employee Attrition dataset and Attrition target
- Status: Superseded provisional decision with confirmed selection
- Context: The assignment requires a tabular dataset with at least 1,000 rows, at least eight feature columns excluding target, and mixed data types suitable for deterministic pipeline validation.
- Decision: Use `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` with target column `Attrition`.
- Consequences: The dataset satisfies assignment size/structure constraints and supports classification with class imbalance (`No`/`Yes`). Because the source dataset is fully clean (0 missing values), deterministic missingness simulation must be introduced in a later phase for assignment-relevant missing-data validation behavior.

## ADR-010: Confirm dataset-specific feature exclusions and audit constraints
- Status: Accepted
- Context: The confirmed dataset includes known non-informative columns that should not be used as predictive features.
- Decision: Exclude `EmployeeNumber` (identifier), `EmployeeCount` (constant), `Over18` (constant), and `StandardHours` (constant) through configuration.
- Consequences: Configuration and future preprocessing logic will enforce deterministic exclusion behavior and reduce leakage/noise risk.

## ADR-011: Use repository-local MLflow tracking for Phase 4
- Status: Accepted
- Context: Phase 4 needs local experiment tracking that works in development and grading without a remote server.
- Decision: Store MLflow runs in the repository-local `mlruns/` directory and opt into the file-based backend when logging or comparing runs.
- Consequences: Experiment artifacts stay local, are ignored by Git, and remain reproducible from the checked-out repository.

## ADR-012: Rank experiments by F1 and exclude failed gates from best-run selection
- Status: Accepted
- Context: The project needs a deterministic, reviewable way to choose the best candidate from several controlled experiments.
- Decision: Rank runs by `f1_attrition` first, then `balanced_accuracy`, `roc_auc`, `precision_attrition`, `recall_attrition`, `accuracy`, run name, and run ID, while excluding failed quality-gate runs from best-run selection.
- Consequences: Comparison output is deterministic, the primary metric remains explicit, and a run that fails the gate cannot become the promoted candidate.

## ADR-013: Use a two-job GitHub Actions CI workflow with isolated MLflow tracking
- Status: Accepted
- Context: Phase 5 needs reproducible CI validation without remote services or repository-local MLflow state leaking into the job output.
- Decision: Run GitHub Actions with separate test and training jobs, cache pip dependencies, set `MLOPS_PIPELINE_MLFLOW_TRACKING_URI` to a temporary path during CI, and use `dvc status` plus raw-data presence checks as the safe non-destructive DVC validation because the repository has no `dvc.yaml` pipeline file.
- Consequences: The workflow stays portable, the training job is gated by successful tests, CI output remains isolated from the repository checkout, and DVC validation is aligned with the repository's current tracking layout.

## ADR-003: Use configuration-driven behavior
- Status: Accepted
- Context: Reproducibility and grader transparency require externalized settings.
- Decision: Keep paths, seeds, split settings, model options, thresholds, and output controls in YAML.
- Consequences: Runtime modules must consume config values and avoid hard-coded business parameters.

## ADR-004: Use explicit MLflow logging
- Status: Accepted
- Context: Assignment evidence requires traceable parameter, metric, data-version, and artifact records.
- Decision: Log MLflow run metadata explicitly rather than relying on implicit behavior.
- Consequences: Experiment modules must include consistent run-tagging and artifact registration logic.

## ADR-005: Use five committed experiment configurations
- Status: Accepted
- Context: Experiment comparison requirements need reproducible and reviewable configuration variants.
- Decision: Maintain five committed experiment YAML files in `configs/experiments/` in later phases.
- Consequences: Configuration review and naming conventions become part of implementation review.

## ADR-006: Use two dependent GitHub Actions jobs
- Status: Accepted
- Context: CI must prevent training execution when tests fail.
- Decision: Implement separate test and training jobs with training dependent on test success.
- Consequences: Workflow logic must encode explicit job dependency and quality gates.

## ADR-007: Use deterministic testing and drift simulation
- Status: Accepted
- Context: Validation and monitoring outputs must be reproducible for grading.
- Decision: Standardize random seeds and deterministic simulation inputs where applicable.
- Consequences: Test fixtures and monitoring checks must avoid non-deterministic behavior.

## ADR-008: Exclude unrequired serving and deployment infrastructure
- Status: Accepted
- Context: Phase scope is tightly bounded to assignment requirements.
- Decision: Exclude FastAPI, Docker, Kubernetes, cloud deployment, orchestration, and feature-store additions.
- Consequences: Engineering effort remains focused on required deliverables and grading criteria.

## ADR-009: Require a grader-accessible final DVC remote
- Status: Accepted
- Context: A local Windows-only remote is not acceptable for external grading execution.
- Decision: Delay final remote setup to a later phase and require a grader-accessible target.
- Consequences: Phase 1 intentionally initializes DVC without binding a final remote.
