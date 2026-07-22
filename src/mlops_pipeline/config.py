"""Configuration loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigError

REQUIRED_TOP_LEVEL_SECTIONS = {
	"project",
	"data",
	"split",
	"features",
	"model",
	"evaluation",
	"mlflow",
	"monitoring",
	"outputs",
}

REQUIRED_SECONDARY_METRICS = {
	"balanced_accuracy",
	"roc_auc",
	"precision_attrition",
	"recall_attrition",
	"accuracy",
}

REQUIRED_MONITORING_OUTPUT_PATHS = {
	"reference_batch_csv",
	"stable_batch_csv",
	"drifted_batch_csv",
	"summary_json",
	"html_report",
}

REQUIRED_MONITORING_DRIFT_MAGNITUDE_KEYS = {
	"numeric_shift",
	"numeric_scale",
	"categorical_constant_share",
}

SUPPORTED_MONITORING_FEATURE_KINDS = {"categorical", "numeric"}
SUPPORTED_MONITORING_TRANSFORMS = {
	"add",
	"scale",
	"set_constant",
}

SUPPORTED_PROJECT_PHASES = {"phase-3", "phase-4", "phase-6"}
SUPPORTED_MODEL_ALGORITHMS = {
	"gradient_boosting",
	"logistic_regression",
	"random_forest",
}

APPROVED_EXCLUDED_FEATURES = {
	"EmployeeNumber",
	"EmployeeCount",
	"Over18",
	"StandardHours",
}

EXPECTED_DATASET_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
EXPECTED_TARGET_COLUMN = "Attrition"


def load_config(config_path: str | Path) -> dict[str, Any]:
	"""Load and validate a YAML config file.

	Args:
		config_path: Path to a YAML configuration file.

	Returns:
		Validated configuration dictionary.

	Raises:
		ConfigError: If loading or validation fails.
	"""
	path = Path(config_path)
	if not path.exists():
		raise ConfigError(f"Configuration file not found: {path}")
	if not path.is_file():
		raise ConfigError(f"Configuration path is not a file: {path}")

	try:
		with path.open("r", encoding="utf-8") as handle:
			loaded = yaml.safe_load(handle)
	except yaml.YAMLError as exc:
		raise ConfigError(f"Malformed YAML in configuration file '{path}': {exc}") from exc
	except OSError as exc:
		raise ConfigError(f"Unable to read configuration file '{path}': {exc}") from exc

	if not isinstance(loaded, dict):
		raise ConfigError("Configuration root must be a mapping/dictionary.")

	validate_config(loaded, config_file_path=path)
	return loaded


def validate_config(config: dict[str, Any], config_file_path: str | Path | None = None) -> None:
	"""Validate full project configuration requirements."""
	missing_sections = REQUIRED_TOP_LEVEL_SECTIONS.difference(config.keys())
	if missing_sections:
		missing = ", ".join(sorted(missing_sections))
		raise ConfigError(f"Configuration is missing required top-level sections: {missing}")

	_require_mapping(config, "project")
	_require_mapping(config, "data")
	_require_mapping(config, "features")

	random_seed = config["project"].get("random_seed")
	if not isinstance(random_seed, int):
		raise ConfigError("'project.random_seed' must be an integer.")

	phase = config["project"].get("phase")
	if not isinstance(phase, str) or not phase.strip():
		raise ConfigError("'project.phase' must be a non-empty string.")
	if phase not in SUPPORTED_PROJECT_PHASES:
		raise ConfigError(
			"'project.phase' must be one of: " + ", ".join(sorted(SUPPORTED_PROJECT_PHASES))
		)

	data_section = config["data"]
	raw_path = data_section.get("raw_path")
	if not isinstance(raw_path, str) or not raw_path.strip():
		raise ConfigError("'data.raw_path' must be a non-empty string.")

	resolved_raw_path = _resolve_dataset_path(raw_path, config_file_path)
	if not resolved_raw_path.exists() or not resolved_raw_path.is_file():
		raise ConfigError(
			"'data.raw_path' must reference an existing dataset file. "
			f"Resolved path: {resolved_raw_path}"
		)
	if resolved_raw_path.name != EXPECTED_DATASET_FILENAME:
		raise ConfigError(
			"'data.raw_path' must point to the verified dataset filename "
			f"'{EXPECTED_DATASET_FILENAME}', got '{resolved_raw_path.name}'."
		)

	target_column = data_section.get("target_column")
	if not isinstance(target_column, str) or not target_column.strip():
		raise ConfigError("'data.target_column' must be a non-empty string.")
	if target_column != EXPECTED_TARGET_COLUMN:
		raise ConfigError(
			f"'data.target_column' must be '{EXPECTED_TARGET_COLUMN}', got '{target_column}'."
		)

	exclude_columns = config["features"].get("exclude_columns")
	if not isinstance(exclude_columns, list) or not all(
		isinstance(item, str) for item in exclude_columns
	):
		raise ConfigError("'features.exclude_columns' must be a list of strings.")

	missing_exclusions = APPROVED_EXCLUDED_FEATURES.difference(set(exclude_columns))
	if missing_exclusions:
		missing = ", ".join(sorted(missing_exclusions))
		raise ConfigError(
			"'features.exclude_columns' must include all approved exclusions: "
			f"{missing}"
		)

	_validate_missingness_config(config["features"])
	_validate_training_config(config)
	_validate_mlflow_config(config["mlflow"])
	_validate_monitoring_config(config["monitoring"])


def _validate_training_config(config: dict[str, Any]) -> None:
	project_phase = config["project"]["phase"]
	split_section = config["split"]
	model_section = config["model"]
	evaluation_section = config["evaluation"]

	test_size = split_section.get("test_size")
	if not isinstance(test_size, (int, float)) or not 0 < float(test_size) < 1:
		raise ConfigError("'split.test_size' must be numeric and in the range (0, 1).")

	random_state = split_section.get("random_state")
	if not isinstance(random_state, int):
		raise ConfigError("'split.random_state' must be an integer.")

	stratify = split_section.get("stratify")
	if not isinstance(stratify, bool):
		raise ConfigError("'split.stratify' must be a boolean.")

	algorithm = model_section.get("algorithm")
	if algorithm not in SUPPORTED_MODEL_ALGORITHMS:
		raise ConfigError(
			"'model.algorithm' must be one of: "
			+ ", ".join(sorted(SUPPORTED_MODEL_ALGORITHMS))
		)

	class_weight = model_section.get("class_weight")
	if project_phase == "phase-3" and algorithm != "logistic_regression":
		raise ConfigError("'model.algorithm' must be 'logistic_regression' for Phase 3.")
	if project_phase == "phase-3" and class_weight != "balanced":
		raise ConfigError("'model.class_weight' must be 'balanced' for Phase 3.")
	if project_phase == "phase-4" and algorithm == "gradient_boosting" and class_weight not in (
		None,
		"",
	):
		raise ConfigError("'model.class_weight' must be null for gradient boosting experiments.")
	if project_phase == "phase-4" and algorithm in {"logistic_regression", "random_forest"}:
		if class_weight in (None, ""):
			pass
		elif isinstance(class_weight, str) and class_weight in {"balanced", "balanced_subsample"}:
			pass
		elif isinstance(class_weight, dict):
			pass
		else:
			raise ConfigError(
				"'model.class_weight' must be null, a supported string, or a mapping for Phase 4."
			)

	max_iter = model_section.get("max_iter")
	if not isinstance(max_iter, int) or max_iter <= 0:
		raise ConfigError("'model.max_iter' must be a positive integer.")

	params = model_section.get("params")
	if params is None:
		model_section["params"] = {}
	elif not isinstance(params, dict):
		raise ConfigError("'model.params' must be a mapping.")

	primary_metric = evaluation_section.get("primary_metric")
	if primary_metric != "f1_attrition":
		raise ConfigError("'evaluation.primary_metric' must be 'f1_attrition'.")

	secondary_metrics = evaluation_section.get("secondary_metrics")
	if not isinstance(secondary_metrics, list) or not all(
		isinstance(metric, str) for metric in secondary_metrics
	):
		raise ConfigError("'evaluation.secondary_metrics' must be a list of strings.")

	missing_metrics = REQUIRED_SECONDARY_METRICS.difference(set(secondary_metrics))
	if missing_metrics:
		raise ConfigError(
			"'evaluation.secondary_metrics' is missing required metrics: "
			+ ", ".join(sorted(missing_metrics))
		)

	thresholds = evaluation_section.get("thresholds")
	if not isinstance(thresholds, dict) or not thresholds:
		raise ConfigError("'evaluation.thresholds' must be a non-empty mapping.")
	if "f1_attrition" not in thresholds:
		raise ConfigError("'evaluation.thresholds' must include 'f1_attrition'.")

	threshold_value = thresholds["f1_attrition"]
	if not isinstance(threshold_value, (int, float)) or not 0 <= float(threshold_value) <= 1:
		raise ConfigError(
			"'evaluation.thresholds.f1_attrition' must be numeric and in the range [0, 1]."
		)


def _validate_mlflow_config(mlflow_section: dict[str, Any]) -> None:
	"""Validate MLflow configuration for local experiment tracking."""
	if not isinstance(mlflow_section, dict):
		raise ConfigError("'mlflow' section must be a mapping/dictionary.")

	tracking_uri = mlflow_section.get("tracking_uri")
	if not isinstance(tracking_uri, str) or not tracking_uri.strip():
		raise ConfigError("'mlflow.tracking_uri' must be a non-empty string.")
	if tracking_uri.startswith("TODO"):
		raise ConfigError("'mlflow.tracking_uri' must be configured for local tracking.")

	experiment_name = mlflow_section.get("experiment_name")
	if not isinstance(experiment_name, str) or not experiment_name.strip():
		raise ConfigError("'mlflow.experiment_name' must be a non-empty string.")
	if experiment_name.startswith("TODO"):
		raise ConfigError("'mlflow.experiment_name' must be configured for Phase 4.")

	log_model = mlflow_section.get("log_model")
	if not isinstance(log_model, bool):
		raise ConfigError("'mlflow.log_model' must be a boolean.")

	run_name = mlflow_section.get("run_name")
	if run_name is not None and (not isinstance(run_name, str) or not run_name.strip()):
		raise ConfigError("'mlflow.run_name' must be a non-empty string when provided.")


def _validate_monitoring_config(monitoring_section: dict[str, Any]) -> None:
	"""Validate Evidently drift-monitoring configuration."""
	if not isinstance(monitoring_section, dict):
		raise ConfigError("'monitoring' section must be a mapping/dictionary.")

	enabled = monitoring_section.get("enabled")
	if not isinstance(enabled, bool):
		raise ConfigError("'monitoring.enabled' must be a boolean.")
	if enabled is False:
		return

	random_seed = monitoring_section.get("random_seed")
	if not isinstance(random_seed, int):
		raise ConfigError("'monitoring.random_seed' must be an integer.")

	reference_batch_size = monitoring_section.get("reference_batch_size")
	if not isinstance(reference_batch_size, int) or reference_batch_size <= 0:
		raise ConfigError("'monitoring.reference_batch_size' must be a positive integer.")

	production_batch_size = monitoring_section.get("production_batch_size")
	if not isinstance(production_batch_size, int) or production_batch_size <= 0:
		raise ConfigError("'monitoring.production_batch_size' must be a positive integer.")

	if reference_batch_size != production_batch_size:
		raise ConfigError(
			"'monitoring.reference_batch_size' and 'monitoring.production_batch_size' must match "
			"for the deterministic reference/current comparison workflow."
		)

	dataset_drift_threshold = monitoring_section.get("dataset_drift_threshold")
	if not isinstance(dataset_drift_threshold, (int, float)) or not 0 <= float(dataset_drift_threshold) <= 1:
		raise ConfigError(
			"'monitoring.dataset_drift_threshold' must be numeric and in the range [0, 1]."
		)

	feature_drift_threshold = monitoring_section.get("feature_drift_threshold")
	if not isinstance(feature_drift_threshold, (int, float)) or not 0 <= float(feature_drift_threshold) <= 1:
		raise ConfigError(
			"'monitoring.feature_drift_threshold' must be numeric and in the range [0, 1]."
		)

	drifted_feature_names = monitoring_section.get("drifted_feature_names")
	if not isinstance(drifted_feature_names, list) or not drifted_feature_names:
		raise ConfigError("'monitoring.drifted_feature_names' must be a non-empty list of strings.")
	if not all(isinstance(item, str) and item.strip() for item in drifted_feature_names):
		raise ConfigError("'monitoring.drifted_feature_names' must contain only non-empty strings.")
	if len(set(drifted_feature_names)) != len(drifted_feature_names):
		raise ConfigError("'monitoring.drifted_feature_names' must not contain duplicates.")

	drift_magnitude = monitoring_section.get("drift_magnitude")
	if not isinstance(drift_magnitude, dict):
		raise ConfigError("'monitoring.drift_magnitude' must be a mapping/dictionary.")
	missing_magnitude_keys = REQUIRED_MONITORING_DRIFT_MAGNITUDE_KEYS.difference(drift_magnitude.keys())
	if missing_magnitude_keys:
		missing = ", ".join(sorted(missing_magnitude_keys))
		raise ConfigError(
			"'monitoring.drift_magnitude' is missing required keys: " f"{missing}"
		)
	for key in REQUIRED_MONITORING_DRIFT_MAGNITUDE_KEYS:
		value = drift_magnitude.get(key)
		if not isinstance(value, (int, float)):
			raise ConfigError(f"'monitoring.drift_magnitude.{key}' must be numeric.")
		if key == "categorical_constant_share" and not 0 < float(value) <= 1:
			raise ConfigError(
				"'monitoring.drift_magnitude.categorical_constant_share' must be in the range (0, 1]."
			)
		if key in {"numeric_shift", "numeric_scale"} and float(value) <= 0:
			raise ConfigError(f"'monitoring.drift_magnitude.{key}' must be greater than 0.")

	per_feature_settings = monitoring_section.get("per_feature_settings")
	if not isinstance(per_feature_settings, dict) or not per_feature_settings:
		raise ConfigError("'monitoring.per_feature_settings' must be a non-empty mapping.")
	if set(per_feature_settings.keys()) != set(drifted_feature_names):
		configured = ", ".join(sorted(per_feature_settings.keys()))
		required = ", ".join(sorted(drifted_feature_names))
		raise ConfigError(
			"'monitoring.per_feature_settings' must define exactly the listed drifted features. "
			f"Configured: {configured}; required: {required}"
		)

	for feature_name, feature_settings in per_feature_settings.items():
		_validate_monitoring_feature_settings(feature_name, feature_settings)

	output_paths = monitoring_section.get("output_paths")
	if not isinstance(output_paths, dict):
		raise ConfigError("'monitoring.output_paths' must be a mapping/dictionary.")
	missing_output_paths = REQUIRED_MONITORING_OUTPUT_PATHS.difference(output_paths.keys())
	if missing_output_paths:
		missing = ", ".join(sorted(missing_output_paths))
		raise ConfigError(
			"'monitoring.output_paths' is missing required keys: " f"{missing}"
		)
	for key in REQUIRED_MONITORING_OUTPUT_PATHS:
		value = output_paths.get(key)
		if not isinstance(value, str) or not value.strip():
			raise ConfigError(f"'monitoring.output_paths.{key}' must be a non-empty string.")


def _validate_monitoring_feature_settings(feature_name: str, feature_settings: Any) -> None:
	if not isinstance(feature_settings, dict):
		raise ConfigError(f"Monitoring settings for '{feature_name}' must be a mapping/dictionary.")

	kind = feature_settings.get("kind")
	if kind not in SUPPORTED_MONITORING_FEATURE_KINDS:
		raise ConfigError(
			f"Monitoring settings for '{feature_name}' must set 'kind' to one of: "
			+ ", ".join(sorted(SUPPORTED_MONITORING_FEATURE_KINDS))
		)

	transform = feature_settings.get("transform")
	if transform not in SUPPORTED_MONITORING_TRANSFORMS:
		raise ConfigError(
			f"Monitoring settings for '{feature_name}' must set 'transform' to one of: "
			+ ", ".join(sorted(SUPPORTED_MONITORING_TRANSFORMS))
		)

	if kind == "numeric":
		magnitude = feature_settings.get("magnitude")
		if not isinstance(magnitude, (int, float)):
			raise ConfigError(f"Numeric monitoring settings for '{feature_name}' must define a numeric 'magnitude'.")
		if transform not in {"add", "scale"}:
			raise ConfigError(
				f"Numeric monitoring settings for '{feature_name}' must use the 'add' or 'scale' transform."
			)
	elif kind == "categorical":
		value = feature_settings.get("value")
		if not isinstance(value, str) or not value.strip():
			raise ConfigError(
				f"Categorical monitoring settings for '{feature_name}' must define a non-empty 'value'."
			)
		if transform != "set_constant":
			raise ConfigError(
				f"Categorical monitoring settings for '{feature_name}' must use the 'set_constant' transform."
			)


def _validate_missingness_config(features_section: dict[str, Any]) -> None:
	missingness = features_section.get("missingness_simulation")
	if not isinstance(missingness, dict):
		raise ConfigError("'features.missingness_simulation' must be a mapping.")

	enabled = missingness.get("enabled")
	if not isinstance(enabled, bool):
		raise ConfigError("'features.missingness_simulation.enabled' must be a boolean.")

	fraction = missingness.get("fraction")
	if not isinstance(fraction, (int, float)):
		raise ConfigError("'features.missingness_simulation.fraction' must be numeric.")
	if not 0 < float(fraction) <= 0.2:
		raise ConfigError(
			"'features.missingness_simulation.fraction' must be in the range (0, 0.2]."
		)

	seed = missingness.get("random_seed")
	if not isinstance(seed, int):
		raise ConfigError("'features.missingness_simulation.random_seed' must be an integer.")

	strategy = missingness.get("strategy")
	columns = missingness.get("columns")
	if strategy not in {"all_eligible", "selected_columns"}:
		raise ConfigError(
			"'features.missingness_simulation.strategy' must be 'all_eligible' or "
			"'selected_columns'."
		)

	if strategy == "selected_columns":
		if not isinstance(columns, list) or not columns:
			raise ConfigError(
				"'features.missingness_simulation.columns' must be a non-empty list when "
				"strategy is 'selected_columns'."
			)
		if not all(isinstance(col, str) and col for col in columns):
			raise ConfigError(
				"'features.missingness_simulation.columns' must contain only non-empty strings."
			)


def _require_mapping(config: dict[str, Any], key: str) -> None:
	value = config.get(key)
	if not isinstance(value, dict):
		raise ConfigError(f"'{key}' section must be a mapping/dictionary.")


def _resolve_dataset_path(raw_path: str, config_file_path: str | Path | None) -> Path:
	candidate = Path(raw_path)
	if candidate.is_absolute():
		return candidate

	cwd_candidate = (Path.cwd() / candidate).resolve()
	if cwd_candidate.exists():
		return cwd_candidate

	if config_file_path is not None:
		config_parent_candidate = (Path(config_file_path).resolve().parent / candidate).resolve()
		if config_parent_candidate.exists():
			return config_parent_candidate

		project_root_candidate = (
			Path(config_file_path).resolve().parent.parent / candidate
		).resolve()
		return project_root_candidate

	return cwd_candidate
