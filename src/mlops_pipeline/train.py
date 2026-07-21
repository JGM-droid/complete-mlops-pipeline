"""Deterministic training orchestration for Phase 3."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import load_config
from .data_validation import DOMAIN_RANGE_CHECKS, validate_dataset
from .evaluate import assess_quality_gate, calculate_classification_metrics
from .exceptions import EvaluationError, QualityGateError, TrainingError
from .phase4_support import (
	build_estimator,
	collect_lineage_metadata,
	log_mlflow_training_run,
)
from .preprocess import (
	build_preprocessor,
	drop_excluded_columns,
	identify_feature_types,
	load_dataframe,
	simulate_missingness,
	split_features_target,
)


@dataclass
class TrainingResult:
	"""Structured result from end-to-end model training."""

	model_pipeline: Pipeline
	metrics: dict[str, Any]
	quality_gate: dict[str, Any]
	train_rows: int
	test_rows: int
	target_distribution_train: dict[str, int]
	target_distribution_test: dict[str, int]
	y_true_test: list[str]
	y_pred_test: list[str]
	y_proba_positive_test: list[float]
	metadata: dict[str, Any]


def train_from_config(config_path: str | Path) -> TrainingResult:
	"""Train and evaluate a deterministic Logistic Regression baseline from config."""
	config = load_config(config_path)
	return train_from_loaded_config(config, config_path=config_path)


def train_from_loaded_config(
	config: dict[str, Any],
	*,
	config_path: str | Path | None = None,
	enforce_quality_gate: bool = True,
) -> TrainingResult:
	"""Train and evaluate a controlled model from loaded config."""
	try:
		data_path = config["data"]["raw_path"]
		target_column = config["data"]["target_column"]
		excluded_columns = list(config["features"]["exclude_columns"])
		split_cfg = config["split"]
		project_seed = int(config["project"]["random_seed"])
		missingness_cfg = config["features"]["missingness_simulation"]
		model_cfg = config["model"]
		eval_cfg = config["evaluation"]
	except KeyError as exc:
		raise TrainingError(f"Required training configuration key is missing: {exc}") from exc

	primary_metric = eval_cfg.get("primary_metric")
	if not isinstance(primary_metric, str) or not primary_metric:
		raise TrainingError("evaluation.primary_metric must be a non-empty string.")
	thresholds = eval_cfg.get("thresholds")
	if not isinstance(thresholds, dict) or not thresholds:
		raise TrainingError("evaluation.thresholds must be a non-empty mapping.")

	df = load_dataframe(data_path)
	required_columns = [
		target_column,
		"EmployeeNumber",
		"EmployeeCount",
		"Over18",
		"StandardHours",
		*DOMAIN_RANGE_CHECKS.keys(),
	]
	validate_dataset(
		df,
		target_column=target_column,
		required_columns=required_columns,
		excluded_columns=excluded_columns,
	)

	X_all, y_all = split_features_target(df, target_column)
	X_all = drop_excluded_columns(X_all, excluded_columns)

	test_size = float(split_cfg["test_size"])
	random_state = int(split_cfg["random_state"])
	use_stratify = bool(split_cfg.get("stratify", True))

	X_train, X_test, y_train, y_test = train_test_split(
		X_all,
		y_all,
		test_size=test_size,
		random_state=random_state,
		stratify=y_all if use_stratify else None,
	)

	X_train_sim, train_missingness_meta = simulate_missingness(
		X_train,
		enabled=bool(missingness_cfg["enabled"]),
		fraction=float(missingness_cfg["fraction"]),
		random_seed=int(missingness_cfg["random_seed"]),
		strategy=str(missingness_cfg["strategy"]),
		columns=missingness_cfg.get("columns"),
	)

	test_seed = int(missingness_cfg["random_seed"]) + 1
	X_test_sim, test_missingness_meta = simulate_missingness(
		X_test,
		enabled=bool(missingness_cfg["enabled"]),
		fraction=float(missingness_cfg["fraction"]),
		random_seed=test_seed,
		strategy=str(missingness_cfg["strategy"]),
		columns=missingness_cfg.get("columns"),
	)

	numeric_cols, categorical_cols = identify_feature_types(X_train_sim)
	preprocessor = build_preprocessor(numeric_cols, categorical_cols)
	model = build_estimator(model_cfg, random_state=project_seed)

	pipeline = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("model", model),
		]
	)

	pipeline.fit(X_train_sim, y_train)

	y_pred = pipeline.predict(X_test_sim)
	model_classes = list(pipeline.named_steps["model"].classes_)
	if "Yes" not in model_classes:
		raise TrainingError(
			"Trained model does not include 'Yes' in class labels; cannot compute "
			"positive-class probabilities."
		)
	positive_idx = model_classes.index("Yes")
	y_proba = pipeline.predict_proba(X_test_sim)[:, positive_idx]

	metrics = calculate_classification_metrics(
		y_true=y_test.to_numpy(),
		y_pred=y_pred,
		y_proba_positive=y_proba,
	)

	quality_gate = assess_quality_gate(
		metrics=metrics,
		thresholds={k: float(v) for k, v in thresholds.items()},
		primary_metric=primary_metric,
	)
	lineage_metadata = collect_lineage_metadata(data_path)
	result = TrainingResult(
		model_pipeline=pipeline,
		metrics=metrics,
		quality_gate=quality_gate,
		train_rows=int(len(X_train_sim)),
		test_rows=int(len(X_test_sim)),
		target_distribution_train={
			str(k): int(v) for k, v in y_train.value_counts().to_dict().items()
		},
		target_distribution_test={
			str(k): int(v) for k, v in y_test.value_counts().to_dict().items()
		},
		y_true_test=[str(v) for v in y_test.to_list()],
		y_pred_test=[str(v) for v in y_pred.tolist()],
		y_proba_positive_test=[float(v) for v in y_proba.tolist()],
		metadata={
			"dataset_path": str(data_path),
			"target_column": target_column,
			"excluded_columns": excluded_columns,
			"model_algorithm": str(model_cfg["algorithm"]),
			"class_weight": model_cfg.get("class_weight"),
			"max_iter": model_cfg.get("max_iter"),
			"split": {
				"test_size": test_size,
				"random_state": random_state,
				"stratify": use_stratify,
			},
			"missingness": {
				"train": train_missingness_meta,
				"test": test_missingness_meta,
			},
			"feature_columns_after_exclusions": X_train_sim.columns.tolist(),
			"positive_class": "Yes",
			"lineage": lineage_metadata,
		},
	)

	try:
		run_id = log_mlflow_training_run(
			config=config,
			config_path=config_path or data_path,
			training_result={
				"metrics": result.metrics,
				"quality_gate": result.quality_gate,
				"metadata": result.metadata,
			},
			model_pipeline=result.model_pipeline,
		)
		result.metadata["mlflow_run_id"] = run_id
	except TrainingError:
		raise
	except Exception as exc:
		raise TrainingError(f"Failed to log MLflow run: {exc}") from exc

	if enforce_quality_gate and not result.quality_gate["passed"]:
		raise QualityGateError(
			"Quality gate failed for metrics: "
			+ ", ".join(
				f"{name}={detail['value']:.4f} < {detail['minimum_required']:.4f}"
				for name, detail in result.quality_gate["failed_metrics"].items()
			)
		)

	return result


def _format_metrics(metrics: dict[str, Any]) -> str:
	display_keys = [
		"f1_attrition",
		"balanced_accuracy",
		"roc_auc",
		"precision_attrition",
		"recall_attrition",
		"accuracy",
	]
	lines = []
	for key in display_keys:
		value = metrics.get(key)
		if isinstance(value, (float, int)):
			lines.append(f"- {key}: {float(value):.4f}")
	return "\n".join(lines)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train and evaluate the deterministic Phase 3 baseline model."
	)
	parser.add_argument(
		"--config",
		required=True,
		help="Path to the YAML config file.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		result = train_from_config(args.config)
	except QualityGateError as exc:
		print("Quality gate status: FAILED")
		print("Action: lower threshold only if justified by stable baseline evidence, "
			  "or improve feature/model settings in-scope.")
		print(f"Reason: {exc}")
		return 1
	except (TrainingError, EvaluationError, ValueError, KeyError) as exc:
		print(f"Training failed: {exc}")
		return 2

	print("Training completed successfully.")
	print(f"Train rows: {result.train_rows}")
	print(f"Test rows: {result.test_rows}")
	print("Metrics:")
	print(_format_metrics(result.metrics))
	print("Quality gate status: PASSED")
	print(
		"Quality gate detail: "
		+ json.dumps(
			{
				"primary_metric": result.quality_gate["primary_metric"],
				"primary_value": result.quality_gate["primary_value"],
				"thresholds": result.quality_gate["thresholds"],
			},
			indent=2,
		)
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
