"""Phase 6 CLI for deterministic Evidently drift monitoring."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from .config import load_config
from .data_validation import DataValidationError, validate_target_column
from .drift_batches import MonitoringContext, generate_monitoring_batches
from .exceptions import ConfigError, MLOpsPipelineError, PreprocessingError
from .preprocess import load_dataframe


@dataclass(slots=True)
class DriftMonitoringResult:
	"""Structured outcome for a drift-monitoring run."""

	reference_path: Path
	current_path: Path
	evaluated_feature_count: int
	drifted_feature_count: int
	drifted_feature_percentage: float
	dataset_drift: bool
	drifted_feature_names: list[str]
	configured_threshold: float
	feature_threshold: float
	gate_status: str
	evidently_version: str
	summary_json_path: Path
	html_report_path: Path
	feature_drift_scores: dict[str, float]


def run_monitoring_from_config(
	config_path: str | Path,
	*,
	current_batch: str = "stable",
) -> DriftMonitoringResult:
	"""Generate deterministic batches and evaluate drift for the chosen current batch."""
	if current_batch not in {"stable", "drifted"}:
		raise MLOpsPipelineError("current_batch must be exactly 'stable' or 'drifted'.")
	config = load_config(config_path)
	batches = generate_monitoring_batches(config, config_path=config_path, persist=True)
	current_path = batches.stable_path if current_batch == "stable" else batches.drifted_path
	return evaluate_drift_from_paths(
		config,
		reference_path=batches.reference_path,
		current_path=current_path,
		config_path=config_path,
	)


def evaluate_drift_from_paths(
	config: dict[str, Any],
	*,
	reference_path: str | Path,
	current_path: str | Path,
	config_path: str | Path | None = None,
) -> DriftMonitoringResult:
	"""Compare reference and current batch files using the configured Evidently preset."""
	context = _build_monitoring_context(config, config_path=config_path)
	reference_dataframe = load_dataframe(reference_path)
	current_dataframe = load_dataframe(current_path)
	_validate_monitoring_batch(reference_dataframe, context, "reference", reference_path)
	_validate_monitoring_batch(current_dataframe, context, "current", current_path)

	reference_features = reference_dataframe[context.feature_columns]
	current_features = current_dataframe[context.feature_columns]
	data_definition = DataDefinition(
		numerical_columns=context.numeric_columns,
		categorical_columns=context.categorical_columns,
	)

	monitoring_section = config["monitoring"]
	feature_threshold = float(monitoring_section["feature_drift_threshold"])
	dataset_threshold = float(monitoring_section["dataset_drift_threshold"])
	report = Report(
		metrics=[
			DataDriftPreset(
				columns=context.feature_columns,
				threshold=feature_threshold,
				drift_share=dataset_threshold,
				include_tests=False,
			),
		]
	)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		snapshot = report.run(
			current_data=Dataset.from_pandas(current_features, data_definition=data_definition),
			reference_data=Dataset.from_pandas(reference_features, data_definition=data_definition),
		)

	metrics = snapshot.dict()["metrics"]
	feature_scores: dict[str, float] = {}
	for metric in metrics:
		metric_name = str(metric.get("metric_name", ""))
		if not metric_name.startswith("ValueDrift("):
			continue
		config_block = metric.get("config", {})
		column_name = config_block.get("column")
		if column_name not in context.feature_columns:
			continue
		feature_scores[str(column_name)] = float(metric["value"])

	drifted_feature_names = [
		feature_name
		for feature_name in context.feature_columns
		if feature_scores.get(feature_name, 1.0) < feature_threshold
	]
	drifted_feature_count = len(drifted_feature_names)
	evaluated_feature_count = len(context.feature_columns)
	drifted_feature_percentage = (
		drifted_feature_count / evaluated_feature_count if evaluated_feature_count else 0.0
	)
	dataset_drift = drifted_feature_percentage >= dataset_threshold
	gate_status = "failed" if dataset_drift else "passed"

	paths = monitoring_section["output_paths"]
	summary_json_path = _resolve_path(paths["summary_json"])
	html_report_path = _resolve_path(paths["html_report"])
	summary_json_path.parent.mkdir(parents=True, exist_ok=True)
	html_report_path.parent.mkdir(parents=True, exist_ok=True)
	_snapshot_to_html(snapshot, html_report_path)

	summary = {
		"reference_path": str(Path(reference_path)),
		"current_path": str(Path(current_path)),
		"evaluated_feature_count": evaluated_feature_count,
		"drifted_feature_count": drifted_feature_count,
		"drifted_feature_percentage": drifted_feature_percentage,
		"dataset_drift": dataset_drift,
		"drifted_feature_names": drifted_feature_names,
		"configured_threshold": dataset_threshold,
		"feature_threshold": feature_threshold,
		"gate_status": gate_status,
		"evidently_version": version("evidently"),
		"feature_drift_scores": feature_scores,
	}
	summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

	return DriftMonitoringResult(
		reference_path=Path(reference_path),
		current_path=Path(current_path),
		evaluated_feature_count=evaluated_feature_count,
		drifted_feature_count=drifted_feature_count,
		drifted_feature_percentage=drifted_feature_percentage,
		dataset_drift=dataset_drift,
		drifted_feature_names=drifted_feature_names,
		configured_threshold=dataset_threshold,
		feature_threshold=feature_threshold,
		gate_status=gate_status,
		evidently_version=version("evidently"),
		summary_json_path=summary_json_path,
		html_report_path=html_report_path,
		feature_drift_scores=feature_scores,
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run deterministic Evidently drift monitoring for the IBM HR attrition dataset."
	)
	parser.add_argument(
		"--config",
		default="configs/train.yaml",
		help="Path to the YAML config file.",
	)
	parser.add_argument(
		"--current-batch",
		choices=("stable", "drifted"),
		default="stable",
		help="Which deterministic generated batch to evaluate against the reference batch.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		result = run_monitoring_from_config(args.config, current_batch=args.current_batch)
	except (ConfigError, DataValidationError, PreprocessingError, ValueError, KeyError) as exc:
		print(f"Drift monitoring failed: {exc}")
		return 2

	print("Drift monitoring completed successfully.")
	print(f"Reference path: {result.reference_path}")
	print(f"Current path: {result.current_path}")
	print(f"Evaluated feature count: {result.evaluated_feature_count}")
	print(f"Drifted feature count: {result.drifted_feature_count}")
	print(f"Drifted feature percentage: {result.drifted_feature_percentage:.4f}")
	print(f"Drifted feature names: {', '.join(result.drifted_feature_names) or '(none)'}")
	print(f"Configured drift threshold: {result.configured_threshold:.4f}")
	print(f"Feature drift threshold: {result.feature_threshold:.4f}")
	print(f"Gate status: {result.gate_status.upper()}")
	print(f"Evidently version: {result.evidently_version}")
	print(f"JSON report: {result.summary_json_path}")
	print(f"HTML report: {result.html_report_path}")
	return 1 if result.dataset_drift else 0


def _build_monitoring_context(config: dict[str, Any], *, config_path: str | Path | None = None) -> MonitoringContext:
	from .drift_batches import build_monitoring_context

	return build_monitoring_context(config, config_path=config_path)


def _validate_monitoring_batch(
	df,
	context: MonitoringContext,
	label: str,
	path: str | Path,
) -> None:
	if list(df.columns) != context.column_order:
		missing = sorted(set(context.column_order).difference(df.columns))
		extra = sorted(set(df.columns).difference(context.column_order))
		raise DataValidationError(
			f"{label.title()} batch schema mismatch for '{path}'. Missing: {missing or 'none'}; "
			f"extra: {extra or 'none'}"
		)
	validate_target_column(df, context.target_column)


def _snapshot_to_html(snapshot, html_path: Path) -> None:
	try:
		snapshot.save_html(str(html_path))
	except AttributeError as exc:
		raise PreprocessingError(f"Evidently snapshot does not support HTML export: {exc}") from exc


def _resolve_path(path_value: str | Path) -> Path:
	path = Path(path_value)
	return path if path.is_absolute() else path.resolve()


if __name__ == "__main__":
	raise SystemExit(main())
