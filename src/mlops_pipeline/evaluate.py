"""Evaluation utilities for Phase 3 classification workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)

from .exceptions import EvaluationError, QualityGateError

POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"


def validate_prediction_inputs(
	y_true: list[str] | np.ndarray,
	y_pred: list[str] | np.ndarray,
	y_proba_positive: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Validate prediction inputs for binary classification evaluation."""
	true_arr = np.asarray(y_true)
	pred_arr = np.asarray(y_pred)
	proba_arr = np.asarray(y_proba_positive, dtype=float)

	if true_arr.ndim != 1 or pred_arr.ndim != 1 or proba_arr.ndim != 1:
		raise EvaluationError("Evaluation inputs must be one-dimensional arrays.")

	if not (len(true_arr) == len(pred_arr) == len(proba_arr)):
		raise EvaluationError(
			"Evaluation inputs must have matching lengths for y_true, y_pred, and "
			"y_proba_positive."
		)

	if len(true_arr) == 0:
		raise EvaluationError("Evaluation inputs cannot be empty.")

	valid_labels = {POSITIVE_LABEL, NEGATIVE_LABEL}
	true_invalid = sorted(set(true_arr.tolist()).difference(valid_labels))
	pred_invalid = sorted(set(pred_arr.tolist()).difference(valid_labels))
	if true_invalid:
		raise EvaluationError(f"y_true contains invalid labels: {true_invalid}")
	if pred_invalid:
		raise EvaluationError(f"y_pred contains invalid labels: {pred_invalid}")

	if np.isnan(proba_arr).any():
		raise EvaluationError("y_proba_positive contains NaN values.")
	if ((proba_arr < 0.0) | (proba_arr > 1.0)).any():
		raise EvaluationError("y_proba_positive values must be in the range [0, 1].")

	return true_arr, pred_arr, proba_arr


def calculate_classification_metrics(
	y_true: list[str] | np.ndarray,
	y_pred: list[str] | np.ndarray,
	y_proba_positive: list[float] | np.ndarray,
) -> dict[str, Any]:
	"""Calculate serializable binary-classification metrics with Yes as positive class."""
	true_arr, pred_arr, proba_arr = validate_prediction_inputs(
		y_true=y_true,
		y_pred=y_pred,
		y_proba_positive=y_proba_positive,
	)

	true_bin = (true_arr == POSITIVE_LABEL).astype(int)

	try:
		roc_auc = float(roc_auc_score(true_bin, proba_arr))
	except ValueError as exc:
		raise EvaluationError(
			"Unable to compute ROC AUC from supplied labels/probabilities."
		) from exc

	metrics: dict[str, Any] = {
		"accuracy": float(accuracy_score(true_arr, pred_arr)),
		"balanced_accuracy": float(balanced_accuracy_score(true_arr, pred_arr)),
		"precision_attrition": float(
			precision_score(true_arr, pred_arr, pos_label=POSITIVE_LABEL, zero_division=0)
		),
		"recall_attrition": float(
			recall_score(true_arr, pred_arr, pos_label=POSITIVE_LABEL, zero_division=0)
		),
		"f1_attrition": float(
			f1_score(true_arr, pred_arr, pos_label=POSITIVE_LABEL, zero_division=0)
		),
		"roc_auc": roc_auc,
	}

	cm = confusion_matrix(true_arr, pred_arr, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
	metrics["confusion_matrix"] = {
		"labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
		"matrix": cm.tolist(),
	}

	return metrics


def assess_quality_gate(
	metrics: dict[str, Any],
	thresholds: dict[str, float],
	*,
	primary_metric: str,
) -> dict[str, Any]:
	"""Assess configured metric thresholds without raising on failure."""
	if primary_metric not in metrics:
		raise EvaluationError(
			f"Primary metric '{primary_metric}' is missing from computed metrics."
		)
	if primary_metric not in thresholds:
		raise EvaluationError(
			f"Primary metric '{primary_metric}' is missing from configured thresholds."
		)

	failed: dict[str, dict[str, float]] = {}
	for metric_name, min_threshold in thresholds.items():
		if metric_name not in metrics:
			raise EvaluationError(
				f"Threshold configured for '{metric_name}', but metric is missing from evaluation output."
			)
		metric_value = metrics[metric_name]
		if not isinstance(metric_value, (int, float)):
			raise EvaluationError(
				f"Metric '{metric_name}' is not numeric and cannot be compared against a threshold."
			)
		if float(metric_value) < float(min_threshold):
			failed[metric_name] = {
				"value": float(metric_value),
				"minimum_required": float(min_threshold),
			}

	return {
		"passed": len(failed) == 0,
		"primary_metric": primary_metric,
		"primary_value": float(metrics[primary_metric]),
		"thresholds": {k: float(v) for k, v in thresholds.items()},
		"failed_metrics": failed,
	}


def evaluate_quality_gate(
	metrics: dict[str, Any],
	thresholds: dict[str, float],
	*,
	primary_metric: str,
) -> dict[str, Any]:
	"""Evaluate configured metric thresholds and raise on failure."""
	result = assess_quality_gate(metrics, thresholds, primary_metric=primary_metric)

	if not result["passed"]:
		raise QualityGateError(
			"Quality gate failed for metrics: "
			+ ", ".join(
				f"{name}={detail['value']:.4f} < {detail['minimum_required']:.4f}"
				for name, detail in result["failed_metrics"].items()
			)
		)

	return result
