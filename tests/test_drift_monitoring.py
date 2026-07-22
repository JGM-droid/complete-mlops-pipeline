"""Tests for Phase 6 deterministic drift monitoring."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mlops_pipeline import monitor_drift
from mlops_pipeline.config import ConfigError, load_config
from mlops_pipeline.data_validation import DataValidationError
from mlops_pipeline.drift_batches import generate_monitoring_batches
from mlops_pipeline.exceptions import PreprocessingError


def _deep_update(target: dict, overrides: dict) -> dict:
	for key, value in overrides.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_update(target[key], value)
		else:
			target[key] = value
	return target


def _write_monitoring_config(base_config_path: Path, tmp_path: Path, overrides: dict | None = None) -> Path:
	with base_config_path.open("r", encoding="utf-8") as handle:
		config = yaml.safe_load(handle)

	config["monitoring"]["output_paths"] = {
		"reference_batch_csv": str(tmp_path / "reference.csv"),
		"stable_batch_csv": str(tmp_path / "stable.csv"),
		"drifted_batch_csv": str(tmp_path / "drifted.csv"),
		"summary_json": str(tmp_path / "summary.json"),
		"html_report": str(tmp_path / "report.html"),
	}
	if overrides:
		_deep_update(config, overrides)

	out_path = tmp_path / "monitoring-config.yaml"
	out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
	return out_path


def _file_hash(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stable_batch_does_not_trigger_dataset_drift(config_path: Path, tmp_path: Path) -> None:
	temp_config = _write_monitoring_config(config_path, tmp_path)
	result = monitor_drift.run_monitoring_from_config(temp_config, current_batch="stable")

	assert result.dataset_drift is False
	assert result.gate_status == "passed"
	assert result.drifted_feature_count == 0
	assert result.drifted_feature_names == []


def test_drifted_batch_triggers_drift_gate(config_path: Path, tmp_path: Path) -> None:
	temp_config = _write_monitoring_config(config_path, tmp_path)
	result = monitor_drift.run_monitoring_from_config(temp_config, current_batch="drifted")

	assert result.dataset_drift is True
	assert result.gate_status == "failed"
	assert result.drifted_feature_count > 0
	assert set(result.drifted_feature_names) == {
		"Age",
		"MonthlyIncome",
		"TotalWorkingYears",
		"JobRole",
		"OverTime",
	}


def test_same_seed_produces_deterministic_batches(config_path: Path, tmp_path: Path) -> None:
	temp_config = _write_monitoring_config(config_path, tmp_path)
	config = load_config(temp_config)
	first = generate_monitoring_batches(config, config_path=temp_config, persist=False)
	second = generate_monitoring_batches(config, config_path=temp_config, persist=False)

	pd.testing.assert_frame_equal(first.reference_batch, second.reference_batch)
	pd.testing.assert_frame_equal(first.stable_batch, second.stable_batch)
	pd.testing.assert_frame_equal(first.drifted_batch, second.drifted_batch)


def test_target_column_is_unchanged_and_excluded_from_mutation(config_path: Path, tmp_path: Path) -> None:
	temp_config = _write_monitoring_config(config_path, tmp_path)
	config = load_config(temp_config)
	batches = generate_monitoring_batches(config, config_path=temp_config, persist=False)

	assert batches.stable_batch["Attrition"].equals(batches.drifted_batch["Attrition"])
	assert not batches.reference_batch.equals(batches.drifted_batch)


def test_json_report_contains_required_fields(config_path: Path, tmp_path: Path) -> None:
	temp_config = _write_monitoring_config(config_path, tmp_path)
	result = monitor_drift.run_monitoring_from_config(temp_config, current_batch="stable")
	report = yaml.safe_load(result.summary_json_path.read_text(encoding="utf-8"))

	for key in [
		"reference_path",
		"current_path",
		"evaluated_feature_count",
		"drifted_feature_count",
		"drifted_feature_percentage",
		"dataset_drift",
		"drifted_feature_names",
		"configured_threshold",
		"gate_status",
		"evidently_version",
	]:
		assert key in report

	assert report["gate_status"] == "passed"
	assert report["evidently_version"] == result.evidently_version


def test_html_report_is_created_and_non_empty(config_path: Path, tmp_path: Path) -> None:
	temp_config = _write_monitoring_config(config_path, tmp_path)
	result = monitor_drift.run_monitoring_from_config(temp_config, current_batch="stable")

	assert result.html_report_path.exists()
	assert result.html_report_path.stat().st_size > 0


def test_missing_input_file_fails_clearly(config_path: Path, tmp_path: Path) -> None:
	temp_config = _write_monitoring_config(config_path, tmp_path)
	config = load_config(temp_config)
	batches = generate_monitoring_batches(config, config_path=temp_config, persist=False)
	missing_current = tmp_path / "missing.csv"

	with pytest.raises(PreprocessingError):
		monitor_drift.evaluate_drift_from_paths(
			config,
			reference_path=batches.reference_path,
			current_path=missing_current,
			config_path=temp_config,
		)


def test_schema_mismatch_fails_clearly(config_path: Path, tmp_path: Path) -> None:
	temp_config = _write_monitoring_config(config_path, tmp_path)
	config = load_config(temp_config)
	batches = generate_monitoring_batches(config, config_path=temp_config, persist=True)
	current_df = pd.read_csv(batches.drifted_path)
	current_df = current_df.rename(columns={"JobRole": "JobRole_Mismatch"})
	current_path = tmp_path / "schema_mismatch.csv"
	current_df.to_csv(current_path, index=False)

	with pytest.raises(DataValidationError):
		monitor_drift.evaluate_drift_from_paths(
			config,
			reference_path=batches.reference_path,
			current_path=current_path,
			config_path=temp_config,
		)


def test_invalid_configuration_fails_clearly(config_path: Path, tmp_path: Path) -> None:
	invalid_config = _write_monitoring_config(
		config_path,
		tmp_path,
		{"monitoring": {"reference_batch_size": 0}},
	)

	with pytest.raises(ConfigError):
		load_config(invalid_config)


def test_cli_returns_expected_success_and_failure_exit_codes(
	config_path: Path, tmp_path: Path, monkeypatch
) -> None:
	temp_config = _write_monitoring_config(config_path, tmp_path)
	monkeypatch.setattr(
		monitor_drift,
		"parse_args",
		lambda: monitor_drift.argparse.Namespace(config=str(temp_config), current_batch="stable"),
	)
	assert monitor_drift.main() == 0

	monkeypatch.setattr(
		monitor_drift,
		"parse_args",
		lambda: monitor_drift.argparse.Namespace(config=str(temp_config), current_batch="drifted"),
	)
	assert monitor_drift.main() == 1


def test_raw_dvc_dataset_remains_byte_for_byte_unchanged(
	config_path: Path, dataset_path: Path, tmp_path: Path
) -> None:
	temp_config = _write_monitoring_config(config_path, tmp_path)
	config = load_config(temp_config)
	initial_hash = _file_hash(dataset_path)
	_ = generate_monitoring_batches(config, config_path=temp_config, persist=False)
	_ = monitor_drift.run_monitoring_from_config(temp_config, current_batch="stable")

	assert _file_hash(dataset_path) == initial_hash


def test_no_paid_or_external_network_calls_occur(config_path: Path, tmp_path: Path, monkeypatch) -> None:
	temp_config = _write_monitoring_config(config_path, tmp_path)
	config = load_config(temp_config)

	def _fail(*args, **kwargs):  # noqa: ANN001
		raise AssertionError("Unexpected external network call")

	monkeypatch.setattr("socket.create_connection", _fail)
	monkeypatch.setattr("urllib.request.urlopen", _fail)
	result = monitor_drift.run_monitoring_from_config(temp_config, current_batch="stable")

	assert result.gate_status == "passed"
