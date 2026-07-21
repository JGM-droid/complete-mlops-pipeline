# Roadmap

## Phase 0: Architecture and planning
- Purpose: Define project scope, architecture principles, controls, and phased execution strategy.
- Principal deliverables: Architecture constitution, roadmap, requirement traceability plan, decision log.
- Acceptance checks: Governing documentation approved and internally consistent.
- Current status: Complete.

## Phase 1: Repository foundation and dataset audit
- Purpose: Build repository baseline, initialize core tooling, and prepare dataset audit capability.
- Principal deliverables: Scaffolding, Git and DVC initialization, dependency compatibility verification, configuration skeleton, dataset audit CLI, README foundation.
- Acceptance checks: Complete. Required structure exists, dependencies import cleanly, dataset audit executed on the selected dataset with evidence recorded, and no out-of-scope implementation was added.
- Current status: Complete.

Phase 1 dataset-audit sub-status:
- Selected dataset confirmed: IBM HR Analytics Employee Attrition.
- Target confirmed: Attrition.
- Audit evidence captured in `reports/dataset_audit.json`.
- Missingness simulation requirement identified for a later phase.

## Phase 2: Configuration, validation, and preprocessing
- Purpose: Implement configuration loader, dataset validation, and preprocessing modules.
- Principal deliverables: Config parsing, schema and quality checks, preprocessing pipeline, initial unit tests.
- Acceptance checks: Complete. Configuration validation, dataset validation, and preprocessing implementation are in place; six-plus preprocessing tests and three-plus dataset-validation tests are implemented and passing.
- Current status: Complete.

## Phase 3: Training and evaluation
- Purpose: Implement deterministic training and evaluation flow.
- Principal deliverables: Training module, evaluation module, model-validation tests, quality gates.
- Acceptance checks: Two model-validation tests and evaluation checks planned and passing.
- Current status: Not started.

## Phase 4: MLflow experiment tracking
- Purpose: Add explicit experiment logging and experiment comparison.
- Principal deliverables: Parameter, metric, data-version, and model-artifact logging; experiment search and comparison.
- Acceptance checks: Five committed experiment configurations executed and compared with `mlflow.search_runs`.
- Current status: Not started.

## Phase 5: Drift monitoring
- Purpose: Implement deterministic data drift monitoring for production data.
- Principal deliverables: Drift summary JSON, HTML report, configurable threshold, deterministic failure behavior.
- Acceptance checks: Monitoring exits with status code 1 on configured drift breach.
- Current status: Not started.

## Phase 6: CI/CD
- Purpose: Implement automated quality and training workflow controls.
- Principal deliverables: GitHub Actions triggers, separate test and training jobs, explicit dependency between jobs.
- Acceptance checks: Green pipeline with training gated by test success and quality checks.
- Current status: Not started.

## Phase 7: Documentation and grader simulation
- Purpose: Finalize operator-facing and grader-facing execution guidance.
- Principal deliverables: Complete README workflow, validation records, simulation notes.
- Acceptance checks: Reproducible local run-through that matches grading expectations.
- Current status: Not started.

## Phase 8: Final audit and submission
- Purpose: Produce final quality audit and submission-ready repository state.
- Principal deliverables: Final requirement traceability update, risk log closure, submission package review.
- Acceptance checks: All mandatory requirements evidenced and no unresolved high-risk gaps.
- Current status: Not started.
