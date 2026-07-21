"""Tests for Phase 3 evaluation logic."""

from __future__ import annotations

import json

import pytest

from mlops_pipeline.evaluate import calculate_classification_metrics, evaluate_quality_gate
from mlops_pipeline.exceptions import EvaluationError, QualityGateError


def _sample_eval_inputs() -> tuple[list[str], list[str], list[float]]:
	# Non-trivial confusion profile to keep metric computations meaningful.
	y_true = ["No", "Yes", "No", "Yes", "No", "Yes", "No", "No"]
	y_pred = ["No", "Yes", "No", "No", "Yes", "Yes", "No", "No"]
	y_proba = [0.10, 0.85, 0.20, 0.40, 0.55, 0.90, 0.15, 0.05]
	return y_true, y_pred, y_proba


def test_metrics_keys_and_serializability() -> None:
	y_true, y_pred, y_proba = _sample_eval_inputs()
	metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

	expected_keys = {
		"accuracy",
		"balanced_accuracy",
		"precision_attrition",
		"recall_attrition",
		"f1_attrition",
		"roc_auc",
		"confusion_matrix",
	}
	assert expected_keys.issubset(metrics.keys())
	json.dumps(metrics)


def test_confusion_matrix_shape_and_content() -> None:
	y_true, y_pred, y_proba = _sample_eval_inputs()
	metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
	cm = metrics["confusion_matrix"]

	assert cm["labels"] == ["No", "Yes"]
	assert len(cm["matrix"]) == 2
	assert len(cm["matrix"][0]) == 2
	assert len(cm["matrix"][1]) == 2
	assert sum(sum(row) for row in cm["matrix"]) == len(y_true)


def test_threshold_pass_behavior() -> None:
	y_true, y_pred, y_proba = _sample_eval_inputs()
	metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

	result = evaluate_quality_gate(
		metrics,
		thresholds={"f1_attrition": 0.30},
		primary_metric="f1_attrition",
	)
	assert result["passed"] is True


def test_threshold_failure_behavior() -> None:
	y_true, y_pred, y_proba = _sample_eval_inputs()
	metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

	with pytest.raises(QualityGateError, match="Quality gate failed"):
		evaluate_quality_gate(
			metrics,
			thresholds={"f1_attrition": 0.95},
			primary_metric="f1_attrition",
		)


def test_invalid_prediction_length_raises_error() -> None:
	y_true, y_pred, y_proba = _sample_eval_inputs()

	with pytest.raises(EvaluationError, match="matching lengths"):
		calculate_classification_metrics(y_true, y_pred[:-1], y_proba)
