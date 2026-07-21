"""Tests for Phase 4 experiment execution and MLflow run comparison."""

from __future__ import annotations

from pathlib import Path

import yaml

from mlops_pipeline.compare_experiments import compare_experiments
from mlops_pipeline.config import load_config
from mlops_pipeline.experiment_runner import discover_experiment_configs, run_experiment_configs
from mlops_pipeline.phase4_support import (
	load_mlflow_runs,
	rank_runs,
	resolve_mlflow_tracking_uri,
	select_best_run,
)
from mlops_pipeline.train import train_from_loaded_config


def _write_temp_config(base_config_path: Path, tmp_path: Path, overrides: dict) -> Path:
	with base_config_path.open("r", encoding="utf-8") as handle:
		cfg = yaml.safe_load(handle)

	for top_level, value in overrides.items():
		if isinstance(value, dict) and isinstance(cfg.get(top_level), dict):
			cfg[top_level].update(value)
		else:
			cfg[top_level] = value

	out = tmp_path / "temp_phase4_comparison.yaml"
	out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
	return out


def test_all_five_controlled_experiments_run_and_are_rankable(repo_root: Path) -> None:
	config_dir = repo_root / "configs" / "experiments"
	config_paths = discover_experiment_configs(config_dir)
	assert len(config_paths) == 5

	results = run_experiment_configs(config_paths)
	assert len(results) == 5
	assert all(item["status"] == "completed" for item in results)

	config = load_config(repo_root / "configs" / "train.yaml")
	tracking_uri = resolve_mlflow_tracking_uri(
		config,
		config_path=repo_root / "configs" / "train.yaml",
	)
	runs = load_mlflow_runs(
		tracking_uri=tracking_uri,
		experiment_name=str(config["mlflow"]["experiment_name"]),
	)
	assert len(runs) >= 5
	ranked = rank_runs(runs)
	assert not ranked.empty
	assert select_best_run(runs) is not None

	comparison = compare_experiments(repo_root / "configs" / "train.yaml")
	assert comparison["best_run"] is not None
	assert comparison["best_run"]["quality_gate_status"] == "passed"


def test_failed_quality_gate_run_cannot_win_best_selection(
	config_path: Path, tmp_path: Path
) -> None:
	pass_result = train_from_loaded_config(
		load_config(config_path),
		config_path=config_path,
		enforce_quality_gate=False,
	)
	failed_cfg = _write_temp_config(
		config_path,
		tmp_path,
		{"evaluation": {"thresholds": {"f1_attrition": 0.99}}},
	)
	failed_result = train_from_loaded_config(
		load_config(failed_cfg),
		config_path=failed_cfg,
		enforce_quality_gate=False,
	)
	assert pass_result.quality_gate["passed"] is True
	assert failed_result.quality_gate["passed"] is False

	runs = load_mlflow_runs(
		tracking_uri=resolve_mlflow_tracking_uri(load_config(config_path), config_path=config_path),
		experiment_name=str(load_config(config_path)["mlflow"]["experiment_name"]),
	)
	best_run = select_best_run(runs)
	assert best_run is not None
	assert best_run["quality_gate_status"] == "passed"