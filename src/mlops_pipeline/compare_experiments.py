"""Phase 4 CLI for ranking MLflow experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .phase4_support import (
	format_comparison_table,
	load_mlflow_runs,
	rank_runs,
	resolve_mlflow_tracking_uri,
	select_best_run,
)


def compare_experiments(config_path: str | Path) -> dict[str, Any]:
	"""Compare completed runs for the configured MLflow experiment."""
	config = load_config(config_path)
	mlflow_section = config["mlflow"]
	tracking_uri = resolve_mlflow_tracking_uri(config, config_path=config_path)
	runs = load_mlflow_runs(
		tracking_uri=tracking_uri,
		experiment_name=str(mlflow_section["experiment_name"]),
	)
	ranked = rank_runs(runs)
	best_run = select_best_run(runs)

	comparison_rows: list[dict[str, Any]] = []
	for _, row in ranked.iterrows():
		comparison_rows.append(
			{
				"run_id": row.get("run_id"),
				"run_name": row.get("run_name"),
				"model_algorithm": row.get("model_algorithm"),
				"quality_gate_status": row.get("quality_gate_status"),
				"f1_attrition": row.get("metric_f1"),
				"balanced_accuracy": row.get("metric_balanced_accuracy"),
				"roc_auc": row.get("metric_roc_auc"),
				"precision_attrition": row.get("metric_precision_attrition"),
				"recall_attrition": row.get("metric_recall_attrition"),
				"accuracy": row.get("metric_accuracy"),
				"important_parameters": {
					"model.class_weight": row.get("params.model.class_weight"),
					"model.max_iter": row.get("params.model.max_iter"),
					"split.test_size": row.get("params.split.test_size"),
					"split.stratify": row.get("params.split.stratify"),
				},
				"eligible": bool(row.get("eligible", False)),
			}
		)

	return {
		"experiment_name": mlflow_section["experiment_name"],
		"tracking_uri": mlflow_section["tracking_uri"],
		"best_run": None if best_run is None else best_run.to_dict(),
		"runs": comparison_rows,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compare controlled Phase 4 MLflow runs.")
	parser.add_argument(
		"--config",
		default="configs/train.yaml",
		help="Config file that defines the MLflow experiment name and tracking URI.",
	)
	parser.add_argument(
		"--output-json",
		default="reports/phase4_experiment_comparison.json",
		help="Where to write the machine-readable comparison summary.",
	)
	parser.add_argument(
		"--output-csv",
		default="reports/phase4_experiment_comparison.csv",
		help="Where to write the machine-readable comparison table.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		comparison = compare_experiments(args.config)
	except Exception as exc:
		print(f"Comparison failed: {exc}")
		return 2

	output_json = Path(args.output_json)
	output_json.parent.mkdir(parents=True, exist_ok=True)
	output_json.write_text(
		json.dumps(comparison, indent=2, sort_keys=True, default=str),
		encoding="utf-8",
	)

	output_csv = Path(args.output_csv)
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	if comparison["runs"]:
		pd.DataFrame(comparison["runs"]).to_csv(output_csv, index=False)
	else:
		output_csv.write_text("", encoding="utf-8")

	loaded_config = load_config(args.config)
	ranked_runs = rank_runs(
		load_mlflow_runs(
			tracking_uri=resolve_mlflow_tracking_uri(loaded_config, config_path=args.config),
			experiment_name=str(loaded_config["mlflow"]["experiment_name"]),
		)
	)
	print(format_comparison_table(ranked_runs))
	if comparison["best_run"] is not None:
		print("Best eligible run:")
		print(json.dumps(comparison["best_run"], indent=2, sort_keys=True, default=str))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
