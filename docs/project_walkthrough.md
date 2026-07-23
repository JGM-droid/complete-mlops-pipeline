# Project Walkthrough

This document is a living engineering reference for the Complete MLOps Pipeline capstone. It records what has actually been built so far and will be updated after each major phase.

## 1. Project Overview

| Item | Summary |
|---|---|
| Business problem | Predict employee attrition so the organization can identify likely departures early and prioritize retention work. |
| Dataset | IBM HR Analytics Employee Attrition dataset. |
| Target variable | `Attrition` with values `Yes` and `No`. |
| Overall pipeline | Dataset audit, configuration validation, dataset validation, preprocessing, deterministic train/test split, model training, MLflow logging, evaluation, quality gate enforcement, experiment comparison, drift monitoring, and CI validation. |

The project is built as a phase-gated pipeline. Early phases established repository controls and dataset audit readiness. Phase 2 added configuration, validation, and preprocessing. Phase 3 added a deterministic baseline model and evaluation path. Phase 4 added MLflow experiment tracking, controlled experiment runs, and deterministic run comparison. Phase 5 added GitHub Actions CI/CD validation. Phase 6 adds deterministic drift monitoring with Evidently. Later phases will add final submission hardening.

## 2. Dataset Summary

| Item | Value |
|---|---|
| Dataset name | IBM HR Analytics Employee Attrition |
| Rows | 1,470 |
| Target column | `Attrition` |
| Train/test split | 80% / 20% |
| Training rows | 1,176 |
| Testing rows | 294 |
| Stratified | Yes |
| Random seed | 42 |

The split happens before preprocessing to avoid leakage. The preprocessing pipeline learns imputers and encoders only from the training fold, then transforms the test fold with those fitted objects. This keeps test metrics honest and prevents statistics from the held-out set from influencing model fit.

## 3. Data Preparation

| Step | What happens | Why |
|---|---|---|
| Target separation | `Attrition` is split from the feature matrix before preprocessing. | The model should not learn from the label column during feature preparation. |
| Explicit exclusions | `EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours` are removed. | `EmployeeNumber` is an identifier; the other three are constant or non-informative. |
| Missingness simulation | Deterministic missing values are injected into features only, using the configured missingness settings. | The raw dataset has no missing values, so this creates a realistic imputation task without touching the target. |
| Numeric imputation | Median imputation. | Median is stable for numeric columns and resists outliers. |
| Categorical imputation | Most-frequent imputation. | This preserves the most common category when values are missing. |
| Encoding strategy | One-hot encoding with `handle_unknown="ignore"`. | This converts categoricals to model-ready columns and remains safe when unseen categories appear later. |
| Target handling | The target is kept separate and is never imputed or altered. | The label must remain trustworthy for training and evaluation. |
| Leakage prevention | The split occurs before preprocessing fit; the preprocessor is fit only on training data. | Training statistics do not leak from the test fold into model preparation. |

## 4. Models Evaluated

| Experiment | Model | Important Parameters | F1 | Recall | Precision | Balanced Accuracy | ROC AUC | Accuracy | Quality Gate | Status | Reason for keeping/rejecting |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| exp-01-logreg-balanced | Logistic Regression | `class_weight=balanced`, `C=1.0`, `max_iter=3000`, `threshold=0.45` | 0.4962 | 0.7021 | 0.3837 | 0.7438 | 0.8103 | 0.7721 | Pass | Kept | Good baseline. Deterministic and gate-compliant. |
| exp-02-logreg-stronger-regularization | Logistic Regression | `class_weight=balanced`, `C=0.5`, `max_iter=3000`, `threshold=0.45` | 0.5075 | 0.7234 | 0.3908 | 0.7544 | 0.8083 | 0.7755 | Pass | Kept | Current best completed model. Highest F1 among eligible runs. |
| exp-03-logreg-no-class_weight | Logistic Regression | `class_weight=null`, `C=1.5`, `max_iter=3000`, `threshold=0.45` | 0.4928 | 0.3617 | 0.7727 | 0.6707 | 0.8141 | 0.8810 | Pass | Rejected | Higher precision and ROC AUC than some runs, but lower F1 and much weaker recall than the current best. |
| exp-04-random-forest-balanced | Random Forest | `class_weight=balanced`, `n_estimators=200`, `max_depth=8`, `min_samples_leaf=2`, `threshold=0.45` | 0.4615 | 0.5106 | 0.4211 | 0.6885 | 0.7735 | 0.8095 | Pass | Rejected | Works as a controlled tree baseline but trails the best Logistic Regression run on F1. |
| exp-05-gradient-boosting | Gradient Boosting | `n_estimators=150`, `learning_rate=0.05`, `subsample=0.9`, `threshold=0.45` | 0.2667 | 0.1702 | 0.6154 | 0.5750 | 0.7696 | 0.8503 | Fail | Rejected | Fails the quality gate and cannot be selected as best. |

## 5. Current Best Model

| Item | Summary |
|---|---|
| Current best model | exp-02-logreg-stronger-regularization |
| Why it is best now | It has the highest `f1_attrition` among eligible runs and still passes the quality gate. |
| Primary metric | `f1_attrition` |
| Current score | 0.5075 |
| Quality gate threshold | 0.45 |
| Quality gate result | Pass |

This section will change as more experiments are completed. A later model may replace the current best if it improves the ranking metric while still passing the gate.

## 6. CI/CD Validation

| Item | Summary |
|---|---|
| Workflow file | `.github/workflows/ci.yml` |
| Triggers | Pushes to `main`, pull requests targeting `main`, and manual workflow dispatch. |
| Test job | Checks out the repository, sets up Python 3.11, installs dependencies, prints dependency versions, runs `compileall`, runs the full `pytest` suite, verifies tracked raw-data files, runs `dvc status`, and checks repository hygiene. |
| Training job | Depends on the test job, repeats environment setup, then runs `python -m mlops_pipeline.train --config configs/train.yaml` with isolated MLflow tracking and checks repository hygiene again. |
| Monitoring job | Depends on the training job, restores the DVC-tracked raw dataset, runs the stable drift check, then runs the deliberate drifted gate check and verifies the nonzero exit without failing the workflow. |
| Merge blockers | Compile failure, test failure, DVC validation failure, baseline training failure, drift-monitoring gate failure, or unexpected repository mutations. |
| Remaining unfinished after this phase | Later submission hardening. GitHub-hosted CI evidence is now recorded in successful run `#29960096801`. |

The workflow is intentionally conservative. It validates the repository without requiring remote credentials, uses the committed raw CSV and `.dvc` pointer, and keeps MLflow tracking in the job's temporary filesystem so CI does not depend on repository-local historical runs.

For the written interpretation of Phase 6 drift outcomes and follow-up actions, see [docs/drift_analysis.md](drift_analysis.md).

## 7. Tests

| Category | What is verified |
|---|---|
| Configuration | The training configuration loads correctly, required sections exist, the selected dataset path resolves, and invalid configuration values fail fast. |
| Data Validation | The raw dataset has the expected columns, valid target values, acceptable numeric ranges, unique employee identifiers, enough rows, and enough usable features after exclusions. |
| Preprocessing | Missing values are simulated deterministically, numeric and categorical imputers work as expected, categorical encoding is safe for unseen categories, and preprocessing does not mutate the original input. |
| Model Validation | The end-to-end training flow returns valid predictions and probabilities, repeated runs are deterministic, excluded columns stay out of the fitted feature set, and a small valid subset can still train successfully. |
| Evaluation | Metric calculation returns the expected classification metrics and confusion matrix, serializes cleanly, and the quality gate passes or fails correctly based on thresholds. |
| MLflow Tracking | The trainer writes run parameters, metrics, tags, artifacts, and the sklearn Pipeline model to the local MLflow store. |
| Experiment Comparison | The comparison utility reads tracked runs, ranks them deterministically, excludes failed-gate runs from selection, and writes machine-readable summaries. |
| CI Behavior | The training CLI exits non-zero when the configured quality gate is impossible, and CI-specific MLflow tracking resolution honors the isolated environment override. |
| Drift Monitoring | The stable batch exits successfully, the deliberately drifted batch fails the configured drift gate, the raw CSV remains byte-for-byte unchanged, and the generated JSON/HTML reports are produced deterministically. |

## 8. Engineering Decisions

- Train/test split exists to create a held-out evaluation set that was not used during model fitting. That is necessary to measure generalization instead of training performance. The split is fixed and deterministic so repeated runs are comparable.

- Stratification was used because the target class is imbalanced. Keeping the same class proportions in train and test makes the evaluation more representative and prevents a lucky or unlucky split from dominating the result.

- Logistic Regression was selected as the baseline because it is fast, deterministic, and easy to inspect. It provides a clean reference point before trying more complex models.

- F1 is the primary metric because the positive class is the one that matters most operationally and the dataset is imbalanced. F1 balances precision and recall, which is more useful here than accuracy alone.

- `class_weight="balanced"` was used to reduce bias toward the majority `No` class. It gives the minority `Yes` class more influence during training without requiring synthetic resampling.

- A quality gate exists so the model cannot be promoted unless it meets a minimum metric threshold. This makes the baseline decision explicit and prevents weaker experiments from being treated as acceptable by default.

- Local MLflow tracking is stored in `mlruns/` for development and experiment comparison, while CI overrides the tracking URI to a temporary location so job output stays isolated from the repository checkout.

- The five experiment set exists to compare small, controlled model changes rather than to run an open-ended search. That keeps the comparison reviewable and lets the experiment table show why one configuration is kept while another is rejected.

- The best-run rule uses `f1_attrition` first and only considers quality-gate-passing runs. That prevents a model with good secondary metrics but poor minority-class balance, or a model that failed the gate, from being promoted.

- Drift monitoring uses a deterministic reference batch plus stable and drifted current batches so that the same seed produces the same reports every time. The stable batch is a row-order permutation of the reference batch, while the drifted batch mutates only feature columns and never `Attrition`.

- The Evidently 0.7.21 API is used through `Report`, `DataDriftPreset`, `DataDefinition`, and `Dataset.from_pandas`. Explicit column typing prevents the HR dataset's string-valued columns from being treated as free-form text.

## 9. Project Timeline

| Phase | What was built | Current status |
|---|---|---|
| Phase 0 | Architecture, planning, and governing documentation. | Complete |
| Phase 1 | Repository foundation, Git/DVC setup, dependency baseline, and dataset audit preparation. | Complete |
| Phase 2 | Configuration loading, dataset validation, preprocessing, and initial unit tests. | Complete |
| Phase 3 | Deterministic training, evaluation, model-validation tests, and quality gate. | Complete |
| Phase 4 | MLflow experiment tracking, controlled experiment runs, and deterministic comparison. | Complete |
| Phase 5 | CI/CD automation. | Complete; validated in GitHub-hosted run `#29960096801` |
| Phase 6 | Drift monitoring. | Complete; stable pass and drifted expected gate-failure validated in run `#29960096801` |
| Phase 7 | Documentation and grader simulation. | Pending |
| Phase 8 | Final audit and submission. | Pending |
