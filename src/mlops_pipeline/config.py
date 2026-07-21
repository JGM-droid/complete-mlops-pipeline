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
	_validate_phase3_training_config(config)


def _validate_phase3_training_config(config: dict[str, Any]) -> None:
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
	if algorithm != "logistic_regression":
		raise ConfigError(
			"'model.algorithm' must be 'logistic_regression' for Phase 3."
		)

	class_weight = model_section.get("class_weight")
	if class_weight != "balanced":
		raise ConfigError("'model.class_weight' must be 'balanced' for Phase 3.")

	max_iter = model_section.get("max_iter")
	if not isinstance(max_iter, int) or max_iter <= 0:
		raise ConfigError("'model.max_iter' must be a positive integer.")

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
