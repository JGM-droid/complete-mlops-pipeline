"""Deterministic batch generation for Phase 6 drift monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import EXPECTED_TARGET_COLUMN
from .data_validation import DOMAIN_RANGE_CHECKS, validate_dataset
from .exceptions import DataValidationError, PreprocessingError
from .preprocess import drop_excluded_columns, identify_feature_types, load_dataframe, split_features_target


@dataclass(slots=True)
class MonitoringContext:
	"""Validated raw-data context used by drift monitoring."""

	raw_dataframe: pd.DataFrame
	target_column: str
	excluded_columns: list[str]
	feature_columns: list[str]
	numeric_columns: list[str]
	categorical_columns: list[str]
	column_order: list[str]


@dataclass(slots=True)
class MonitoringBatches:
	"""Persisted reference, stable, and drifted batches."""

	context: MonitoringContext
	reference_batch: pd.DataFrame
	stable_batch: pd.DataFrame
	drifted_batch: pd.DataFrame
	reference_path: Path
	stable_path: Path
	drifted_path: Path


def build_monitoring_context(config: dict[str, Any], *, config_path: str | Path | None = None) -> MonitoringContext:
	"""Load and validate the canonical raw dataset for drift monitoring."""
	try:
		raw_path = config["data"]["raw_path"]
		target_column = str(config["data"]["target_column"])
		excluded_columns = list(config["features"]["exclude_columns"])
	except KeyError as exc:
		raise PreprocessingError(f"Monitoring configuration is missing required keys: {exc}") from exc

	raw_dataframe = load_dataframe(_resolve_path(raw_path))
	required_columns = [
		target_column,
		*DOMAIN_RANGE_CHECKS.keys(),
		"EmployeeCount",
		"EmployeeNumber",
		"Over18",
		"StandardHours",
	]
	validate_dataset(
		raw_dataframe,
		target_column=target_column,
		required_columns=required_columns,
		excluded_columns=excluded_columns,
	)

	feature_matrix, _ = split_features_target(raw_dataframe, target_column)
	feature_matrix = drop_excluded_columns(feature_matrix, excluded_columns)
	numeric_columns, categorical_columns = identify_feature_types(feature_matrix)

	return MonitoringContext(
		raw_dataframe=raw_dataframe,
		target_column=target_column,
		excluded_columns=excluded_columns,
		feature_columns=feature_matrix.columns.tolist(),
		numeric_columns=numeric_columns,
		categorical_columns=categorical_columns,
		column_order=raw_dataframe.columns.tolist(),
	)


def generate_monitoring_batches(
	config: dict[str, Any],
	*,
	config_path: str | Path | None = None,
	persist: bool = True,
) -> MonitoringBatches:
	"""Create deterministic reference, stable, and drifted batches."""
	context = build_monitoring_context(config, config_path=config_path)
	monitoring = config["monitoring"]
	seed = int(monitoring["random_seed"])
	reference_batch_size = int(monitoring["reference_batch_size"])
	production_batch_size = int(monitoring["production_batch_size"])
	if reference_batch_size != production_batch_size:
		raise DataValidationError(
			"Monitoring batch generation requires matching reference and production batch sizes."
		)

	if reference_batch_size > len(context.raw_dataframe):
		raise DataValidationError(
			"Monitoring batch size cannot exceed the number of available raw dataset rows."
		)

	rng = np.random.RandomState(seed)
	reference_indices = rng.choice(len(context.raw_dataframe), size=reference_batch_size, replace=False)
	reference_batch = (
		context.raw_dataframe.iloc[np.sort(reference_indices)].copy(deep=True).reset_index(drop=True)
	)
	stable_batch = reference_batch.sample(frac=1, random_state=seed + 1).reset_index(drop=True)
	drifted_batch = _apply_drift_to_batch(
		stable_batch,
		monitoring["per_feature_settings"],
		monitoring["drift_magnitude"],
	)

	if not drifted_batch[context.target_column].equals(stable_batch[context.target_column]):
		raise DataValidationError("Monitoring drift simulation must not modify the target column.")

	output_paths = monitoring["output_paths"]
	reference_path = _resolve_path(output_paths["reference_batch_csv"])
	stable_path = _resolve_path(output_paths["stable_batch_csv"])
	drifted_path = _resolve_path(output_paths["drifted_batch_csv"])

	if persist:
		for output_path in (reference_path, stable_path, drifted_path):
			output_path.parent.mkdir(parents=True, exist_ok=True)
		reference_batch.to_csv(reference_path, index=False)
		stable_batch.to_csv(stable_path, index=False)
		drifted_batch.to_csv(drifted_path, index=False)

	return MonitoringBatches(
		context=context,
		reference_batch=reference_batch,
		stable_batch=stable_batch,
		drifted_batch=drifted_batch,
		reference_path=reference_path,
		stable_path=stable_path,
		drifted_path=drifted_path,
	)


def _apply_drift_to_batch(
	batch: pd.DataFrame,
	per_feature_settings: dict[str, Any],
	drift_magnitude: dict[str, Any],
) -> pd.DataFrame:
	drifted = batch.copy(deep=True)
	for feature_name, feature_settings in per_feature_settings.items():
		if feature_name not in drifted.columns:
			raise DataValidationError(
				f"Configured drift feature '{feature_name}' is missing from the monitoring batch schema."
			)

		kind = feature_settings["kind"]
		transform = feature_settings["transform"]
		if kind == "numeric":
			magnitude = float(feature_settings["magnitude"])
			series = pd.to_numeric(drifted[feature_name], errors="raise")
			if transform == "add":
				series = series + magnitude
			elif transform == "scale":
				series = series * magnitude
			else:
				raise DataValidationError(
					f"Unsupported numeric transform '{transform}' for feature '{feature_name}'."
				)
			series = _apply_optional_numeric_clip(series, feature_settings)
			if pd.api.types.is_integer_dtype(drifted[feature_name]):
				series = series.round().astype(drifted[feature_name].dtype)
			drifted[feature_name] = series
		elif kind == "categorical":
			if transform != "set_constant":
				raise DataValidationError(
					f"Unsupported categorical transform '{transform}' for feature '{feature_name}'."
				)
			drifted[feature_name] = feature_settings["value"]
		else:
			raise DataValidationError(
				f"Unsupported drift feature kind '{kind}' for feature '{feature_name}'."
			)

	# Preserve the target column exactly as-is.
	if EXPECTED_TARGET_COLUMN in drifted.columns:
		drifted[EXPECTED_TARGET_COLUMN] = batch[EXPECTED_TARGET_COLUMN].copy(deep=True)

	# The top-level drift magnitude is validated and retained for documentation/traceability.
	_ = drift_magnitude
	return drifted


def _apply_optional_numeric_clip(series: pd.Series, feature_settings: dict[str, Any]) -> pd.Series:
	clip_min = feature_settings.get("clip_min")
	clip_max = feature_settings.get("clip_max")
	if clip_min is not None:
		series = series.clip(lower=clip_min)
	if clip_max is not None:
		series = series.clip(upper=clip_max)
	return series


def _resolve_path(path_value: str | Path) -> Path:
	path = Path(path_value)
	return path if path.is_absolute() else path.resolve()