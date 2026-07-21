"""Tests for Phase 4 MLflow tracking and model validation behavior."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
import yaml

from mlops_pipeline.config import load_config
from mlops_pipeline.exceptions import ConfigError
from mlops_pipeline.phase4_support import build_estimator
from mlops_pipeline.train import train_from_config, train_from_loaded_config


def _write_temp_config(base_config_path: Path, tmp_path: Path, overrides: dict) -> Path:
	with base_config_path.open("r", encoding="utf-8") as handle:
		cfg = yaml.safe_load(handle)

	for top_level, value in overrides.items():
		if isinstance(value, dict) and isinstance(cfg.get(top_level), dict):
			cfg[top_level].update(value)
		else:
			cfg[top_level] = value

	out = tmp_path / "temp_phase4.yaml"
	out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
	return out


def test_mlflow_run_creation_and_required_artifacts(config_path: Path) -> None:
	result = train_from_config(config_path)
	config = load_config(config_path)
	client = mlflow.tracking.MlflowClient()
	run_id = result.metadata["mlflow_run_id"]

	assert run_id
	assert result.quality_gate["passed"] is True

	run = client.get_run(run_id)
	assert run.data.params["model.algorithm"] == "logistic_regression"
	assert run.data.params["model.class_weight"] == "balanced"
	assert run.data.metrics["f1_attrition"] >= 0.0
	assert run.data.tags["quality_gate.status"] == "passed"

	assert [item.path for item in client.list_artifacts(run_id, path="model")]
	assert [item.path for item in client.list_artifacts(run_id, path="config")]
	assert [item.path for item in client.list_artifacts(run_id, path="evaluation")]
	assert [item.path for item in client.list_artifacts(run_id, path="metadata")]

	experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
	assert experiment is not None


def test_supported_model_types_build_valid_estimators() -> None:
	assert build_estimator(
		{"algorithm": "logistic_regression", "class_weight": "balanced", "max_iter": 100},
		random_state=42,
	).__class__.__name__ == "LogisticRegression"
	assert build_estimator(
		{"algorithm": "random_forest", "class_weight": "balanced", "params": {"n_estimators": 10}},
		random_state=42,
	).__class__.__name__ == "RandomForestClassifier"
	assert build_estimator(
		{"algorithm": "gradient_boosting", "class_weight": None, "params": {"n_estimators": 10}},
		random_state=42,
	).__class__.__name__ == "GradientBoostingClassifier"


def test_unknown_model_type_raises_expected_error(config_path: Path, tmp_path: Path) -> None:
	with config_path.open("r", encoding="utf-8") as handle:
		cfg = yaml.safe_load(handle)
	cfg["project"]["phase"] = "phase-4"
	cfg["model"]["algorithm"] = "unsupported_model"
	temp_cfg = tmp_path / "bad_model.yaml"
	temp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

	with pytest.raises(ConfigError, match="model.algorithm"):
		load_config(temp_cfg)


def test_deterministic_repeated_runs_remain_reproducible(config_path: Path) -> None:
	r1 = train_from_config(config_path)
	r2 = train_from_config(config_path)
	assert r1.metrics == r2.metrics
	assert r1.y_pred_test == r2.y_pred_test
	assert r1.y_proba_positive_test == r2.y_proba_positive_test


def test_missing_git_or_dvc_metadata_fails_safely(monkeypatch: pytest.MonkeyPatch, config_path: Path) -> None:
	monkeypatch.setattr("mlops_pipeline.phase4_support._run_command", lambda *args, **kwargs: None)
	result = train_from_config(config_path)
	assert result.metadata["lineage"]["git_commit"] == "unavailable"
	assert result.metadata["lineage"]["dvc_status"] == "unavailable"
	assert result.quality_gate["passed"] is True


def test_quality_gate_status_logged_even_when_failed(config_path: Path, tmp_path: Path) -> None:
	failed_cfg = _write_temp_config(
		config_path,
		tmp_path,
		{"evaluation": {"thresholds": {"f1_attrition": 0.99}}},
	)
	result = train_from_loaded_config(
		load_config(failed_cfg),
		config_path=failed_cfg,
		enforce_quality_gate=False,
	)
	assert result.quality_gate["passed"] is False
	assert result.metadata["mlflow_run_id"]