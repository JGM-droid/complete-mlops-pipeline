"""Tests for Phase 3 deterministic model training and validation behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from mlops_pipeline.exceptions import QualityGateError
from mlops_pipeline.train import train_from_config


def _write_temp_config(base_config_path: Path, tmp_path: Path, overrides: dict) -> Path:
	with base_config_path.open("r", encoding="utf-8") as handle:
		cfg = yaml.safe_load(handle)

	for top_level, value in overrides.items():
		if isinstance(value, dict) and isinstance(cfg.get(top_level), dict):
			cfg[top_level].update(value)
		else:
			cfg[top_level] = value

	out = tmp_path / "temp_train.yaml"
	out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
	return out


def test_predictions_shape_type_length_and_allowed_values(config_path: Path) -> None:
	result = train_from_config(config_path)

	assert isinstance(result.y_pred_test, list)
	assert isinstance(result.y_true_test, list)
	assert isinstance(result.y_proba_positive_test, list)
	assert len(result.y_pred_test) == result.test_rows
	assert len(result.y_true_test) == result.test_rows
	assert len(result.y_proba_positive_test) == result.test_rows
	assert set(result.y_pred_test).issubset({"Yes", "No"})


def test_probability_output_range_and_length(config_path: Path) -> None:
	result = train_from_config(config_path)
	assert len(result.y_proba_positive_test) == result.test_rows
	assert all(0.0 <= p <= 1.0 for p in result.y_proba_positive_test)


def test_quality_gate_passes_for_default_config(config_path: Path) -> None:
	result = train_from_config(config_path)
	assert result.quality_gate["passed"] is True
	assert result.metrics["f1_attrition"] >= result.quality_gate["thresholds"]["f1_attrition"]


def test_quality_gate_failure_with_temp_threshold_override(
	config_path: Path, tmp_path: Path
) -> None:
	temp_cfg = _write_temp_config(
		config_path,
		tmp_path,
		{"evaluation": {"thresholds": {"f1_attrition": 0.99}}},
	)

	with pytest.raises(QualityGateError):
		train_from_config(temp_cfg)


def test_deterministic_repeatability_same_seed(config_path: Path) -> None:
	r1 = train_from_config(config_path)
	r2 = train_from_config(config_path)

	assert r1.metrics == r2.metrics
	assert r1.y_pred_test == r2.y_pred_test
	assert r1.y_proba_positive_test == r2.y_proba_positive_test


def test_target_absent_from_fitted_feature_names(config_path: Path) -> None:
	result = train_from_config(config_path)
	feature_names = result.model_pipeline.named_steps["preprocessor"].get_feature_names_out()
	feature_blob = " ".join(feature_names.tolist())

	assert "Attrition" not in feature_blob
	assert "EmployeeNumber" not in feature_blob
	assert "EmployeeCount" not in feature_blob
	assert "Over18" not in feature_blob
	assert "StandardHours" not in feature_blob


def test_training_on_deterministic_subset_still_produces_valid_predictions(
	raw_dataset_df: pd.DataFrame, config_path: Path, tmp_path: Path
) -> None:
	# Keep >=1000 rows to satisfy validation while reducing test runtime.
	subset = raw_dataset_df.head(1100).copy(deep=True)
	subset_path = tmp_path / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
	subset.to_csv(subset_path, index=False)

	temp_cfg = _write_temp_config(
		config_path,
		tmp_path,
		{
			"data": {"raw_path": str(subset_path)},
			"evaluation": {"thresholds": {"f1_attrition": 0.0}},
		},
	)

	result = train_from_config(temp_cfg)
	assert len(result.y_pred_test) == result.test_rows
	assert set(result.y_pred_test).issubset({"Yes", "No"})
