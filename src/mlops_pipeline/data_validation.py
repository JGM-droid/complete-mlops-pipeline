"""Dataset schema and quality validation utilities for Phase 2."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .exceptions import DataValidationError

TARGET_ALLOWED_VALUES = {"Yes", "No"}
EXPECTED_CONSTANT_COLUMNS = {"EmployeeCount", "Over18", "StandardHours"}
MINIMUM_ROW_COUNT = 1000
MINIMUM_USABLE_FEATURES = 8

DOMAIN_RANGE_CHECKS = {
	"Age": (18, 60),
	"DistanceFromHome": (1, 29),
	"PercentSalaryHike": (11, 25),
	"TotalWorkingYears": (0, 60),
	"YearsAtCompany": (0, 60),
}


def validate_dataset(
	df: pd.DataFrame,
	*,
	target_column: str,
	required_columns: Iterable[str],
	excluded_columns: Iterable[str],
	max_duplicate_rows: int = 0,
) -> None:
	"""Run full dataset validation checks for Phase 2.

	Raises:
		DataValidationError: If any validation rule fails.
	"""
	validate_dataframe(df)
	validate_required_columns(df, required_columns)
	validate_target_column(df, target_column)
	validate_numeric_ranges(df, DOMAIN_RANGE_CHECKS)
	validate_expected_constants(df)
	validate_employee_number_uniqueness(df)
	validate_duplicate_rows(df, max_duplicate_rows=max_duplicate_rows)
	validate_minimum_rows(df, minimum_rows=MINIMUM_ROW_COUNT)
	validate_usable_feature_count(
		df,
		target_column=target_column,
		excluded_columns=list(excluded_columns),
		minimum_features=MINIMUM_USABLE_FEATURES,
	)


def validate_dataframe(df: pd.DataFrame) -> None:
	"""Validate DataFrame type and non-empty state."""
	if not isinstance(df, pd.DataFrame):
		raise DataValidationError(
			f"Input dataset must be a pandas DataFrame, got {type(df).__name__}."
		)
	if df.empty:
		raise DataValidationError("Input DataFrame is empty.")


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
	"""Validate required column presence."""
	missing = sorted(set(required_columns).difference(df.columns))
	if missing:
		raise DataValidationError(f"Dataset is missing required columns: {', '.join(missing)}")


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
	"""Validate target existence, missingness, and allowed values."""
	if target_column not in df.columns:
		raise DataValidationError(f"Target column '{target_column}' does not exist.")

	target = df[target_column]
	if target.isna().any():
		missing_count = int(target.isna().sum())
		raise DataValidationError(
			f"Target column '{target_column}' contains missing values: {missing_count}"
		)

	unique_values = set(target.unique().tolist())
	invalid = sorted(unique_values.difference(TARGET_ALLOWED_VALUES))
	if invalid:
		raise DataValidationError(
			f"Target column '{target_column}' contains invalid values: {invalid}. "
			f"Expected only: {sorted(TARGET_ALLOWED_VALUES)}"
		)


def validate_numeric_ranges(
	df: pd.DataFrame, range_rules: dict[str, tuple[float, float]]
) -> None:
	"""Validate reasonable numeric ranges for selected stable columns."""
	for column, (min_allowed, max_allowed) in range_rules.items():
		if column not in df.columns:
			raise DataValidationError(f"Range check column missing: '{column}'")
		if not pd.api.types.is_numeric_dtype(df[column]):
			raise DataValidationError(f"Range check column must be numeric: '{column}'")

		out_of_range = df[(df[column] < min_allowed) | (df[column] > max_allowed)]
		if not out_of_range.empty:
			raise DataValidationError(
				f"Column '{column}' has values outside [{min_allowed}, {max_allowed}]."
			)


def validate_expected_constants(df: pd.DataFrame) -> None:
	"""Validate that known constant columns remain constant."""
	for column in EXPECTED_CONSTANT_COLUMNS:
		if column not in df.columns:
			raise DataValidationError(f"Expected constant column missing: '{column}'")
		unique_count = int(df[column].nunique(dropna=False))
		if unique_count != 1:
			raise DataValidationError(
				f"Expected constant column '{column}' has {unique_count} unique values."
			)


def validate_employee_number_uniqueness(df: pd.DataFrame) -> None:
	"""Validate uniqueness of EmployeeNumber identifier."""
	column = "EmployeeNumber"
	if column not in df.columns:
		raise DataValidationError(f"Expected identifier column missing: '{column}'")
	duplicate_identifiers = int(df[column].duplicated().sum())
	if duplicate_identifiers > 0:
		raise DataValidationError(
			f"Identifier column '{column}' contains duplicates: {duplicate_identifiers}"
		)


def validate_duplicate_rows(df: pd.DataFrame, max_duplicate_rows: int = 0) -> None:
	"""Validate duplicate-row count threshold."""
	duplicate_rows = int(df.duplicated().sum())
	if duplicate_rows > max_duplicate_rows:
		raise DataValidationError(
			f"Dataset has {duplicate_rows} duplicate rows; maximum allowed is "
			f"{max_duplicate_rows}."
		)


def validate_minimum_rows(df: pd.DataFrame, minimum_rows: int = MINIMUM_ROW_COUNT) -> None:
	"""Validate minimum dataset size."""
	row_count = int(df.shape[0])
	if row_count < minimum_rows:
		raise DataValidationError(
			f"Dataset must have at least {minimum_rows} rows, found {row_count}."
		)


def validate_usable_feature_count(
	df: pd.DataFrame,
	*,
	target_column: str,
	excluded_columns: Iterable[str],
	minimum_features: int = MINIMUM_USABLE_FEATURES,
) -> None:
	"""Validate minimum number of usable feature columns."""
	feature_columns = [
		col for col in df.columns if col != target_column and col not in set(excluded_columns)
	]
	usable_count = len(feature_columns)
	if usable_count < minimum_features:
		raise DataValidationError(
			f"Dataset must provide at least {minimum_features} usable features, "
			f"found {usable_count}."
		)
