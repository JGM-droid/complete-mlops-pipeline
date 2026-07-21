# Project Walkthrough

This document is a living engineering reference for the Complete MLOps Pipeline capstone. It records what has actually been built so far and will be updated after each major phase.

## 1. Project Overview

| Item | Summary |
|---|---|
| Business problem | Predict employee attrition so the organization can identify likely departures early and prioritize retention work. |
| Dataset | IBM HR Analytics Employee Attrition dataset. |
| Target variable | `Attrition` with values `Yes` and `No`. |
| Overall pipeline | Dataset audit, configuration validation, dataset validation, preprocessing, deterministic train/test split, model training, evaluation, and quality gate enforcement. |

The project is built as a phase-gated pipeline. Early phases established repository controls and dataset audit readiness. Phase 2 added configuration, validation, and preprocessing. Phase 3 added a deterministic baseline model and evaluation path. Later phases will add experiment tracking, drift monitoring, CI/CD, and final submission hardening.

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
| Phase 3 baseline | Logistic Regression | `class_weight=balanced`, `max_iter=3000`, `solver=liblinear`, `random_state=42`, `threshold=0.45` | 0.4962 | 0.7021 | 0.3837 | 0.7438 | 0.8103 | 0.7721 | Pass | Kept | Current best completed model. It is deterministic, passes the configured gate, and provides a solid minority-class recall baseline. |
| Pending experiment 2 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Pending experiment 3 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

## 5. Current Best Model

| Item | Summary |
|---|---|
| Current best model | Logistic Regression baseline |
| Why it is best now | It is the only completed experiment so far, it is deterministic across repeated runs, and it satisfies the quality gate. |
| Primary metric | `f1_attrition` |
| Current score | 0.4962 |
| Quality gate threshold | 0.45 |
| Quality gate result | Pass |

This section will change as more experiments are completed. Once a stronger model is trained and validated, this summary will be updated to reflect the new best candidate.

## 6. Tests

| Category | What is verified |
|---|---|
| Configuration | The training configuration loads correctly, required sections exist, the selected dataset path resolves, and invalid configuration values fail fast. |
| Data Validation | The raw dataset has the expected columns, valid target values, acceptable numeric ranges, unique employee identifiers, enough rows, and enough usable features after exclusions. |
| Preprocessing | Missing values are simulated deterministically, numeric and categorical imputers work as expected, categorical encoding is safe for unseen categories, and preprocessing does not mutate the original input. |
| Model Validation | The end-to-end training flow returns valid predictions and probabilities, repeated runs are deterministic, excluded columns stay out of the fitted feature set, and a small valid subset can still train successfully. |
| Evaluation | Metric calculation returns the expected classification metrics and confusion matrix, serializes cleanly, and the quality gate passes or fails correctly based on thresholds. |

## 7. Engineering Decisions

- Train/test split exists to create a held-out evaluation set that was not used during model fitting. That is necessary to measure generalization instead of training performance. The split is fixed and deterministic so repeated runs are comparable.

- Stratification was used because the target class is imbalanced. Keeping the same class proportions in train and test makes the evaluation more representative and prevents a lucky or unlucky split from dominating the result.

- Logistic Regression was selected as the baseline because it is fast, deterministic, and easy to inspect. It provides a clean reference point before trying more complex models.

- F1 is the primary metric because the positive class is the one that matters most operationally and the dataset is imbalanced. F1 balances precision and recall, which is more useful here than accuracy alone.

- `class_weight="balanced"` was used to reduce bias toward the majority `No` class. It gives the minority `Yes` class more influence during training without requiring synthetic resampling.

- A quality gate exists so the model cannot be promoted unless it meets a minimum metric threshold. This makes the baseline decision explicit and prevents weaker experiments from being treated as acceptable by default.

## 8. Project Timeline

| Phase | What was built | Current status |
|---|---|---|
| Phase 0 | Architecture, planning, and governing documentation. | Complete |
| Phase 1 | Repository foundation, Git/DVC setup, dependency baseline, and dataset audit preparation. | Complete |
| Phase 2 | Configuration loading, dataset validation, preprocessing, and initial unit tests. | Complete |
| Phase 3 | Deterministic training, evaluation, model-validation tests, and quality gate. | Complete |
| Phase 4 | MLflow experiment tracking and comparison. | Pending |
| Phase 5 | Drift monitoring. | Pending |
| Phase 6 | CI/CD automation. | Pending |
| Phase 7 | Documentation and grader simulation. | Pending |
| Phase 8 | Final audit and submission. | Pending |
