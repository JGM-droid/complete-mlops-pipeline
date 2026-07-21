"""Tests for dataset validation rules."""

from __future__ import annotations

import pandas as pd
import pytest

from mlops_pipeline.data_validation import (
	validate_dataset,
	validate_numeric_ranges,
	validate_required_columns,
	validate_target_column,
)
from mlops_pipeline.exceptions import DataValidationError


def test_expected_columns_are_present(
	raw_dataset_df: pd.DataFrame, required_columns: list[str]
) -> None:
	validate_required_columns(raw_dataset_df, required_columns)


def test_attrition_contains_only_yes_and_no(
	raw_dataset_df: pd.DataFrame, target_column: str
) -> None:
	validate_target_column(raw_dataset_df, target_column)


def test_selected_numeric_features_within_reasonable_ranges(
	raw_dataset_df: pd.DataFrame,
) -> None:
	validate_numeric_ranges(
		raw_dataset_df,
		{
			"Age": (18, 60),
			"DistanceFromHome": (1, 29),
			"PercentSalaryHike": (11, 25),
			"TotalWorkingYears": (0, 60),
			"YearsAtCompany": (0, 60),
		},
	)


def test_full_dataset_validation_passes(
	raw_dataset_df: pd.DataFrame,
	target_column: str,
	required_columns: list[str],
	excluded_columns: list[str],
) -> None:
	validate_dataset(
		raw_dataset_df,
		target_column=target_column,
		required_columns=required_columns,
		excluded_columns=excluded_columns,
	)


def test_invalid_target_value_raises_error() -> None:
	df = pd.DataFrame(
		{
			"Age": [30, 40],
			"Attrition": ["No", "Maybe"],
			"DistanceFromHome": [5, 3],
			"EmployeeCount": [1, 1],
			"EmployeeNumber": [1, 2],
			"Over18": ["Y", "Y"],
			"PercentSalaryHike": [12, 15],
			"StandardHours": [80, 80],
			"TotalWorkingYears": [5, 10],
			"YearsAtCompany": [2, 4],
			"Department": ["Sales", "Research & Development"],
			"JobRole": ["Sales Executive", "Research Scientist"],
			"MonthlyIncome": [3000, 6000],
			"BusinessTravel": ["Travel_Rarely", "Non-Travel"],
			"DailyRate": [1000, 1200],
		}
	)
	with pytest.raises(DataValidationError, match="invalid values"):
		validate_target_column(df, "Attrition")


def test_non_unique_employee_number_raises_error(
	raw_dataset_df: pd.DataFrame,
	target_column: str,
	required_columns: list[str],
	excluded_columns: list[str],
) -> None:
	df = raw_dataset_df.copy(deep=True)
	df.loc[df.index[1], "EmployeeNumber"] = df.loc[df.index[0], "EmployeeNumber"]

	with pytest.raises(DataValidationError, match="contains duplicates"):
		validate_dataset(
			df,
			target_column=target_column,
			required_columns=required_columns,
			excluded_columns=excluded_columns,
		)


def test_too_few_rows_raises_error(
	raw_dataset_df: pd.DataFrame,
	target_column: str,
	required_columns: list[str],
	excluded_columns: list[str],
) -> None:
	df = raw_dataset_df.head(100).copy(deep=True)

	with pytest.raises(DataValidationError, match="at least 1000 rows"):
		validate_dataset(
			df,
			target_column=target_column,
			required_columns=required_columns,
			excluded_columns=excluded_columns,
		)
