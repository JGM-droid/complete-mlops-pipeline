"""Phase 4 CLI for running controlled MLflow experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .exceptions import MLOpsPipelineError
from .train import train_from_loaded_config


def discover_experiment_configs(config_dir: str | Path) -> list[Path]:
	"""Discover experiment YAML files in deterministic order."""
	path = Path(config_dir)
	if not path.exists() or not path.is_dir():
		raise MLOpsPipelineError(f"Experiment config directory is invalid: {path}")
	return sorted(p for p in path.glob("*.yaml") if p.is_file())


def run_experiment_configs(config_paths: list[Path]) -> list[dict[str, Any]]:
	"""Run each controlled experiment and collect a structured summary."""
	results: list[dict[str, Any]] = []
	for config_path in config_paths:
		config = load_config(config_path)
		try:
			training_result = train_from_loaded_config(
				config,
				config_path=config_path,
				enforce_quality_gate=False,
			)
		except Exception as exc:
			results.append(
				{
					"config_path": str(config_path),
					"status": "failed",
					"error": str(exc),
				}
			)
			continue

		results.append(
			{
				"config_path": str(config_path),
				"status": "completed",
				"run_id": training_result.metadata.get("mlflow_run_id"),
				"quality_gate_passed": training_result.quality_gate["passed"],
				"primary_metric": training_result.quality_gate["primary_metric"],
				"primary_value": training_result.quality_gate["primary_value"],
				"metrics": training_result.metrics,
			},
		)
	return results


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run the controlled Phase 4 MLflow experiment set."
	)
	parser.add_argument(
		"--config-dir",
		default="configs/experiments",
		help="Directory containing the five controlled experiment YAML files.",
	)
	parser.add_argument(
		"--output-json",
		default="reports/phase4_experiment_runs.json",
		help="Where to write the machine-readable experiment summary.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		config_paths = discover_experiment_configs(args.config_dir)
		results = run_experiment_configs(config_paths)
	except Exception as exc:
		print(f"Experiment execution failed: {exc}")
		return 2

	output_path = Path(args.output_json)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

	print("Experiment execution summary:")
	print(json.dumps(results, indent=2, sort_keys=True))
	if any(item.get("status") != "completed" for item in results):
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
