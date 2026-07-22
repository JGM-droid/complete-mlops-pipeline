"""Tests for Phase 5 CI-critical behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mlops_pipeline import train
from mlops_pipeline.config import load_config
from mlops_pipeline.phase4_support import resolve_mlflow_tracking_uri


def _write_temp_config(base_config_path: Path, tmp_path: Path, overrides: dict) -> Path:
	with base_config_path.open("r", encoding="utf-8") as handle:
		cfg = yaml.safe_load(handle)

	for top_level, value in overrides.items():
		if isinstance(value, dict) and isinstance(cfg.get(top_level), dict):
			cfg[top_level].update(value)
		else:
			cfg[top_level] = value

	out = tmp_path / "temp_ci.yaml"
	out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
	return out


def test_train_main_returns_nonzero_when_quality_gate_is_impossible(
	config_path: Path, tmp_path: Path, monkeypatch
) -> None:
	temp_cfg = _write_temp_config(
		config_path,
		tmp_path,
		{"evaluation": {"thresholds": {"f1_attrition": 0.99}}},
	)
	monkeypatch.setattr(train, "parse_args", lambda: argparse.Namespace(config=str(temp_cfg)))

	assert train.main() == 1


def test_mlflow_tracking_uri_override_uses_isolated_location(
	config_path: Path, tmp_path: Path, monkeypatch
) -> None:
	tracking_dir = tmp_path / "mlflow-ci"
	monkeypatch.setenv("MLOPS_PIPELINE_MLFLOW_TRACKING_URI", tracking_dir.as_uri())

	config = load_config(config_path)
	assert resolve_mlflow_tracking_uri(config, config_path=config_path) == tracking_dir.as_uri()