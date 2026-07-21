"""Phase 4 MLflow, experiment-comparison, and model-factory helpers."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from numbers import Real
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .exceptions import TrainingError

SUPPORTED_MODEL_ALGORITHMS = {
	"gradient_boosting",
	"logistic_regression",
	"random_forest",
}

MLFLOW_TRACKING_URI_ENV_VAR = "MLOPS_PIPELINE_MLFLOW_TRACKING_URI"


@dataclass(frozen=True)
class RankedRun:
	run_id: str
	run_name: str
	status: str
	model_algorithm: str
	quality_gate_status: str
	f1_attrition: float | None
	balanced_accuracy: float | None
	roc_auc: float | None
	precision_attrition: float | None
	recall_attrition: float | None
	accuracy: float | None
	important_parameters: dict[str, Any]
	eligible: bool


def build_estimator(model_section: dict[str, Any], *, random_state: int) -> Any:
	"""Build a supported scikit-learn estimator from a model config block."""
	if not isinstance(model_section, dict):
		raise TrainingError("model section must be a mapping.")

	algorithm = model_section.get("algorithm")
	if algorithm not in SUPPORTED_MODEL_ALGORITHMS:
		raise TrainingError(
			"Unsupported model.algorithm. Supported values are: "
			+ ", ".join(sorted(SUPPORTED_MODEL_ALGORITHMS))
		)

	params = model_section.get("params") or {}
	if not isinstance(params, dict):
		raise TrainingError("model.params must be a mapping when provided.")

	if algorithm == "logistic_regression":
		reserved = {"random_state", "class_weight", "max_iter", "solver"}
		conflicts = reserved.intersection(params)
		if conflicts:
			raise TrainingError(
				"model.params contains reserved keys managed by the project config: "
				+ ", ".join(sorted(conflicts))
			)
		max_iter = model_section.get("max_iter")
		if not isinstance(max_iter, int) or max_iter <= 0:
			raise TrainingError("model.max_iter must be a positive integer.")
		return LogisticRegression(
			class_weight=model_section.get("class_weight"),
			max_iter=max_iter,
			random_state=random_state,
			solver="liblinear",
			**params,
		)

	if algorithm == "random_forest":
		reserved = {"random_state", "class_weight"}
		conflicts = reserved.intersection(params)
		if conflicts:
			raise TrainingError(
				"model.params contains reserved keys managed by the project config: "
				+ ", ".join(sorted(conflicts))
			)
		return RandomForestClassifier(
			class_weight=model_section.get("class_weight"),
			random_state=random_state,
			n_jobs=1,
			**params,
		)

	if model_section.get("class_weight") not in (None, "", {}):
		raise TrainingError("Gradient boosting does not support class_weight in this project.")
	reserved = {"random_state"}
	conflicts = reserved.intersection(params)
	if conflicts:
		raise TrainingError(
			"model.params contains reserved keys managed by the project config: "
			+ ", ".join(sorted(conflicts))
		)
	return GradientBoostingClassifier(random_state=random_state, **params)


def resolve_mlflow_tracking_uri(config: dict[str, Any], config_path: str | Path | None = None) -> str:
	"""Resolve the MLflow tracking URI with a test-safe override."""
	override = os.getenv(MLFLOW_TRACKING_URI_ENV_VAR)
	if override:
		return override

	mlflow_section = config.get("mlflow")
	if not isinstance(mlflow_section, dict):
		raise TrainingError("mlflow section must be a mapping.")

	raw_uri = mlflow_section.get("tracking_uri")
	if not isinstance(raw_uri, str) or not raw_uri.strip():
		raise TrainingError("mlflow.tracking_uri must be a non-empty string.")

	raw_uri = raw_uri.strip()
	if raw_uri.startswith(("file:", "sqlite:", "http://", "https://")):
		return raw_uri

	path = Path(raw_uri)
	if not path.is_absolute():
		path = (Path.cwd() / path).resolve()
	return path.as_uri()


def collect_lineage_metadata(dataset_path: str | Path) -> dict[str, Any]:
	"""Collect Git and DVC metadata while failing safely when unavailable."""
	repo_root = Path(__file__).resolve().parents[2]
	dataset_path = Path(dataset_path)
	git_commit = _run_command(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
	git_status = _run_command(["git", "status", "--short"], cwd=repo_root)
	dvc_status = _run_command(["dvc", "status", "--json"], cwd=repo_root)

	return {
		"dataset_path": str(dataset_path),
		"dataset_name": dataset_path.name,
		"git_available": git_commit is not None,
		"git_commit": git_commit or "unavailable",
		"git_dirty": bool(git_status and git_status.strip()),
		"git_status": git_status or "unavailable",
		"dvc_available": dvc_status is not None,
		"dvc_status": dvc_status or "unavailable",
	}


def log_mlflow_training_run(
	*,
	config: dict[str, Any],
	config_path: str | Path,
	training_result: dict[str, Any],
	model_pipeline: Any,
) -> str:
	"""Log a completed training result to the configured local MLflow store."""
	mlflow_section = config.get("mlflow")
	if not isinstance(mlflow_section, dict):
		raise TrainingError("mlflow section must be a mapping.")

	tracking_uri = resolve_mlflow_tracking_uri(config, config_path=config_path)
	experiment_name = mlflow_section.get("experiment_name")
	if not isinstance(experiment_name, str) or not experiment_name.strip():
		raise TrainingError("mlflow.experiment_name must be a non-empty string.")
	experiment_name = experiment_name.strip()

	run_name = mlflow_section.get("run_name")
	if not isinstance(run_name, str) or not run_name.strip():
		run_name = f"{config['model']['algorithm']}-{config['project']['phase']}"
	else:
		run_name = run_name.strip()

	_allow_file_store(tracking_uri)
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(experiment_name)

	with mlflow.start_run(run_name=run_name) as run:
		mlflow.log_params(_build_logged_params(config))
		mlflow.log_metrics(
			{
				key: float(value)
				for key, value in training_result["metrics"].items()
				if key != "confusion_matrix" and isinstance(value, (int, float))
			}
		)

		with tempfile.TemporaryDirectory() as tmp_dir:
			tmp_path = Path(tmp_dir)
			evaluation_path = tmp_path / "evaluation_results.json"
			evaluation_path.write_text(
				json.dumps(training_result["metrics"], indent=2, sort_keys=True),
				encoding="utf-8",
			)
			quality_gate_path = tmp_path / "quality_gate.json"
			quality_gate_path.write_text(
				json.dumps(training_result["quality_gate"], indent=2, sort_keys=True),
				encoding="utf-8",
			)
			confusion_matrix_path = tmp_path / "confusion_matrix.json"
			confusion_matrix_path.write_text(
				json.dumps(training_result["metrics"]["confusion_matrix"], indent=2, sort_keys=True),
				encoding="utf-8",
			)
			lineage_path = tmp_path / "lineage.json"
			lineage_path.write_text(
				json.dumps(training_result["metadata"].get("lineage", {}), indent=2, sort_keys=True),
				encoding="utf-8",
			)
			mlflow.log_artifact(str(evaluation_path), artifact_path="evaluation")
			mlflow.log_artifact(str(quality_gate_path), artifact_path="evaluation")
			mlflow.log_artifact(str(confusion_matrix_path), artifact_path="evaluation")
			mlflow.log_artifact(str(lineage_path), artifact_path="metadata")

		config_source = Path(config_path)
		if config_source.exists() and config_source.is_file():
			mlflow.log_artifact(str(config_source), artifact_path="config")

		if bool(mlflow_section.get("log_model", True)):
			mlflow.sklearn.log_model(
				model_pipeline,
				artifact_path="model",
				serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
			)

		mlflow.set_tag("project.phase", str(config["project"]["phase"]))
		mlflow.set_tag("dataset.name", Path(config["data"]["raw_path"]).name)
		mlflow.set_tag("target.column", str(config["data"]["target_column"]))
		mlflow.set_tag(
			"quality_gate.status",
			"passed" if training_result["quality_gate"]["passed"] else "failed",
		)
		mlflow.set_tag("source.git_commit", training_result["metadata"]["lineage"].get("git_commit", "unavailable"))
		mlflow.set_tag("source.dvc_status", training_result["metadata"]["lineage"].get("dvc_status", "unavailable"))
		mlflow.set_tag("mlflow.experiment_name", experiment_name)
		mlflow.set_tag("mlflow.run_name", run_name)

		return run.info.run_id


def load_mlflow_runs(*, tracking_uri: str, experiment_name: str) -> pd.DataFrame:
	"""Load runs for an MLflow experiment using the local tracking store."""
	_allow_file_store(tracking_uri)
	mlflow.set_tracking_uri(tracking_uri)
	experiment = mlflow.get_experiment_by_name(experiment_name)
	if experiment is None:
		return pd.DataFrame()

	runs = mlflow.search_runs(
		experiment_ids=[experiment.experiment_id],
		output_format="pandas",
		order_by=["metrics.f1_attrition DESC", "attributes.start_time DESC"],
	)
	if runs.empty:
		return runs

	runs = runs.copy(deep=True)
	runs["quality_gate_status"] = runs.get("tags.quality_gate.status")
	runs["run_name"] = runs.get("tags.mlflow.runName")
	runs["model_algorithm"] = runs.get("params.model.algorithm")
	runs["eligible"] = runs["quality_gate_status"].fillna("failed").eq("passed")
	runs["_sort_run_name"] = runs["run_name"].fillna("")
	runs["_sort_run_id"] = runs["run_id"].fillna("")
	return runs


def rank_runs(runs: pd.DataFrame) -> pd.DataFrame:
	"""Rank runs by the primary metric with deterministic tie-breaking."""
	if runs.empty:
		return runs

	working = runs.copy(deep=True)
	working["metric_f1"] = pd.to_numeric(working.get("metrics.f1_attrition"), errors="coerce")
	working["metric_balanced_accuracy"] = pd.to_numeric(
		working.get("metrics.balanced_accuracy"), errors="coerce"
	)
	working["metric_roc_auc"] = pd.to_numeric(working.get("metrics.roc_auc"), errors="coerce")
	working["metric_precision_attrition"] = pd.to_numeric(
		working.get("metrics.precision_attrition"), errors="coerce"
	)
	working["metric_recall_attrition"] = pd.to_numeric(
		working.get("metrics.recall_attrition"), errors="coerce"
	)
	working["metric_accuracy"] = pd.to_numeric(working.get("metrics.accuracy"), errors="coerce")

	eligible = working[working["eligible"] == True].copy(deep=True)  # noqa: E712
	if eligible.empty:
		eligible = working.copy(deep=True)

	return eligible.sort_values(
		by=[
			"metric_f1",
			"metric_balanced_accuracy",
			"metric_roc_auc",
			"metric_precision_attrition",
			"metric_recall_attrition",
			"metric_accuracy",
			"_sort_run_name",
			"_sort_run_id",
		],
		ascending=[False, False, False, False, False, False, True, True],
		na_position="last",
	)


def select_best_run(runs: pd.DataFrame) -> pd.Series | None:
	"""Select the best eligible run, or None if no runs exist."""
	ranked = rank_runs(runs)
	if ranked.empty:
		return None
	return ranked.iloc[0]


def format_comparison_table(runs: pd.DataFrame) -> str:
	"""Render a human-readable markdown comparison table."""
	if runs.empty:
		return "No MLflow runs were found."

	columns = [
		"run_name",
		"run_id",
		"model_algorithm",
		"quality_gate_status",
		"metric_f1",
		"metric_recall_attrition",
		"metric_precision_attrition",
		"metric_balanced_accuracy",
		"metric_roc_auc",
		"metric_accuracy",
	]
	labels = {
		"run_name": "Run Name",
		"run_id": "Run ID",
		"model_algorithm": "Model Type",
		"quality_gate_status": "Quality Gate",
		"metric_f1": "F1",
		"metric_recall_attrition": "Recall",
		"metric_precision_attrition": "Precision",
		"metric_balanced_accuracy": "Balanced Accuracy",
		"metric_roc_auc": "ROC AUC",
		"metric_accuracy": "Accuracy",
	}
	lines = [
		"| " + " | ".join(labels[column] for column in columns) + " |",
		"|" + "|".join(["---"] * len(columns)) + "|",
	]
	for _, row in runs.iterrows():
		values: list[str] = []
		for column in columns:
			value = row.get(column)
			if value is None or pd.isna(value):
				if column == "metric_f1":
					value = row.get("f1_attrition")
				elif column == "metric_balanced_accuracy":
					value = row.get("balanced_accuracy")
				elif column == "metric_roc_auc":
					value = row.get("roc_auc")
				elif column == "metric_precision_attrition":
					value = row.get("precision_attrition")
				elif column == "metric_recall_attrition":
					value = row.get("recall_attrition")
				elif column == "metric_accuracy":
					value = row.get("accuracy")
			if column.startswith("metric_") and isinstance(value, Real):
				values.append(f"{float(value):.4f}")
			else:
				values.append(str(value) if value is not None else "")
		lines.append("| " + " | ".join(values) + " |")
	return "\n".join(lines)


def _build_logged_params(config: dict[str, Any]) -> dict[str, Any]:
	model_section = config["model"]
	features_section = config["features"]
	split_section = config["split"]
	evaluation_section = config["evaluation"]
	missingness = features_section.get("missingness_simulation", {})
	model_params = model_section.get("params") or {}

	params: dict[str, Any] = {
		"project.phase": config["project"]["phase"],
		"project.random_seed": config["project"]["random_seed"],
		"model.algorithm": model_section.get("algorithm"),
		"model.class_weight": model_section.get("class_weight"),
		"model.max_iter": model_section.get("max_iter"),
		"split.test_size": split_section.get("test_size"),
		"split.random_state": split_section.get("random_state"),
		"split.stratify": split_section.get("stratify"),
		"evaluation.primary_metric": evaluation_section.get("primary_metric"),
		"evaluation.threshold.f1_attrition": evaluation_section.get("thresholds", {}).get(
			"f1_attrition"
		),
		"features.missingness.enabled": missingness.get("enabled"),
		"features.missingness.fraction": missingness.get("fraction"),
		"features.missingness.random_seed": missingness.get("random_seed"),
		"features.missingness.strategy": missingness.get("strategy"),
		"features.exclude_columns": ",".join(features_section.get("exclude_columns", [])),
	}

	for key, value in model_params.items():
		params[f"model.param.{key}"] = value

	return params


def _run_command(command: list[str], *, cwd: Path) -> str | None:
	try:
		completed = subprocess.run(
			command,
			cwd=cwd,
			capture_output=True,
			text=True,
			check=True,
		)
	except (FileNotFoundError, subprocess.CalledProcessError):
		return None

	stdout = completed.stdout.strip()
	return stdout or None


def _allow_file_store(tracking_uri: str) -> None:
	"""Opt in to the local MLflow file store when the tracking URI is file-based."""
	if tracking_uri.startswith("file:") or "://" not in tracking_uri:
		os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")