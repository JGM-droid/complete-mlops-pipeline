# Architecture Constitution

## Project purpose
This repository establishes a complete MLOps pipeline for a tabular employee-attrition prediction use case, beginning with a controlled, phase-gated foundation and expanding through validated implementation stages.

## TripleTen success criteria
- A reproducible repository structure with clear separation of concerns.
- Versioned datasets and data lineage evidence.
- Configuration-driven behavior across training, evaluation, and monitoring.
- Deterministic validation and testing workflows.
- Traceable experiment and model comparison workflow.
- Documented CI checks and monitoring outputs aligned with assignment requirements.

## Architecture principles
- Architecture-first delivery: implementation must follow approved architecture.
- Phase-gate workflow: each phase has explicit acceptance checks.
- Configuration-first design: operational parameters are externalized in YAML.
- Modular business logic: small, focused modules in a `src` package layout.
- Reproducibility and traceability: code, data, config, and metrics must be linked.
- Change-control process: architecture-impacting changes require documentation updates before implementation.

## Repository structure
- `src/mlops_pipeline/`: core package modules.
- `configs/`: train and experiment configuration files.
- `data/`: raw, processed, and production datasets.
- `tests/`: unit and validation test suites by concern.
- `docs/`: governing documentation and architecture records.
- `reports/`: generated audit and monitoring outputs.

## Module ownership
- Project Owner: approves scope, phase completion, and release decisions.
- Architecture review: verifies conformance to governing documentation.
- Implementation review: verifies code quality, test readiness, and requirement traceability.

## Planned data flow
1. Ingest versioned raw dataset from `data/raw/`.
2. Validate dataset schema and quality constraints.
3. Apply preprocessing to create model-ready features.
4. Train configured model on train split.
5. Evaluate model with configured metrics and quality gates.
6. Log experiments and artifacts for reproducible comparison.
7. Monitor production-data drift and issue deterministic alerts.

## Configuration-first philosophy
All configurable paths, random seeds, model settings, thresholds, and experiment settings are defined in YAML and consumed by runtime modules. Hard-coded business settings are prohibited except for temporary Phase 1 placeholders explicitly marked as TODO.

## Data lineage concept
Required lineage linkage for each production-relevant run:
- Git commit
- DVC data hash
- Configuration snapshot
- MLflow run
- Model artifact
- Evaluation metrics

## Technology stack
- Python 3.11 (preferred and currently selected)
- pandas and scikit-learn for tabular workflows
- PyYAML for configuration loading
- DVC for dataset versioning
- MLflow for experiment tracking (planned phases)
- pytest for tests (planned phases)
- GitHub Actions for CI (planned phases)
- Evidently for deterministic drift reporting (planned phases)

## Testing layers
- Unit tests for preprocessing and configuration behavior.
- Dataset-validation tests for schema and integrity checks.
- Model-validation tests for quality gates and constraints.
- Evaluation tests for metric calculation and threshold handling.
- Drift-monitoring tests for deterministic change detection behavior.

## CI design
Planned CI uses GitHub Actions with separate test and training jobs. The training job depends on successful test completion and enforces a performance quality gate before artifact promotion.

## Monitoring design
Planned monitoring runs deterministic drift checks using configurable thresholds, writes machine-readable summaries, emits an HTML report, and returns non-zero exit status when drift exceeds policy.

## Phase-gate workflow
- A phase begins only after governing documentation is aligned.
- Deliverables are implemented only for the active phase scope.
- Acceptance checks must pass before promoting to the next phase.
- Conflicts with approved architecture are reported before implementation.

## Definition of done
A phase is complete when:
- Required deliverables exist and satisfy acceptance checks.
- Requirement traceability is updated.
- Risks and unresolved items are documented.
- Architecture and decision logs reflect approved changes.

## Explicitly excluded scope
The following are excluded unless explicitly approved in a later phase:
- Serving APIs and web backends
- Docker, Kubernetes, or orchestration infrastructure
- Cloud deployment design
- Feature store integration
- Non-required platform expansion

## Locked architecture decisions
1. Use Python 3.11 if it is installed and available.
2. Use standard src layout `src/mlops_pipeline/`.
3. Use the IBM HR Analytics Employee Attrition dataset with source file `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` and target `Attrition`.
4. Use pandas and scikit-learn for tabular modeling.
5. Use YAML configuration through PyYAML.
6. Use DVC for dataset versioning.
7. Use MLflow later for explicit experiment tracking.
8. Use pytest later for unit, data-validation, and model-validation tests.
9. Use GitHub Actions later with separate test and training jobs.
10. Use Evidently later for deterministic drift monitoring.
11. All configurable paths, seeds, model settings, thresholds, and experiment settings come from YAML.
12. Business logic is separated into small modules.
13. No lesson code is copied.
14. Do not add unrequired serving or deployment infrastructure.
15. Do not configure a final DVC remote in Phase 1.
16. Do not create duplicate implementations or wrappers containing business logic.

## Confirmed dataset audit facts
- Dataset: IBM HR Analytics Employee Attrition
- Source filename: `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Target column: `Attrition`
- Rows: 1,470
- Columns: 35 total (34 excluding target)
- Numeric and categorical columns are both present
- Target classes: `No` and `Yes`
- Class distribution: `No` = 1,233 (83.8776%), `Yes` = 237 (16.1224%)
- Duplicate rows: 0
- Missing values: 0
- Assignment implication: deterministic missingness simulation is required in a later phase

## Conflict reporting rule
Any architectural conflict, ambiguity, or unsafe implementation request must be reported and resolved through governing documentation before implementation continues.
