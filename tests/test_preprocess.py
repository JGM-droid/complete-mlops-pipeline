"""Tests for preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_pipeline.exceptions import PreprocessingError
from mlops_pipeline.preprocess import (
	build_preprocessor,
	drop_excluded_columns,
	identify_feature_types,
	load_dataframe,
	simulate_missingness,
	split_features_target,
	transform_features,
)


def _sample_frame() -> pd.DataFrame:
	return pd.DataFrame(
		{
			"Age": [30, 40, 35, 29],
			"DistanceFromHome": [5, 9, 2, 3],
			"BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel", "Travel_Rarely"],
			"Department": ["Sales", "Research & Development", "Human Resources", "Sales"],
			"Attrition": ["No", "Yes", "No", "No"],
			"EmployeeNumber": [1, 2, 3, 4],
			"EmployeeCount": [1, 1, 1, 1],
			"Over18": ["Y", "Y", "Y", "Y"],
			"StandardHours": [80, 80, 80, 80],
		}
	)


def test_missing_numeric_values_are_imputed() -> None:
	df = _sample_frame()
	df.loc[1, "Age"] = np.nan

	X, _ = split_features_target(df, "Attrition")
	X = drop_excluded_columns(X, ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])
	numeric_cols, categorical_cols = identify_feature_types(X)
	preprocessor = build_preprocessor(numeric_cols, categorical_cols)
	transformed = transform_features(X, preprocessor, fit=True)

	assert np.isnan(transformed).sum() == 0


def test_missing_categorical_values_are_imputed() -> None:
	df = _sample_frame()
	df.loc[2, "Department"] = np.nan

	X, _ = split_features_target(df, "Attrition")
	X = drop_excluded_columns(X, ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])
	numeric_cols, categorical_cols = identify_feature_types(X)
	preprocessor = build_preprocessor(numeric_cols, categorical_cols)
	transformed = transform_features(X, preprocessor, fit=True)

	assert np.isnan(transformed).sum() == 0


def test_categorical_variables_are_encoded() -> None:
	df = _sample_frame()
	X, _ = split_features_target(df, "Attrition")
	X = drop_excluded_columns(X, ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])

	numeric_cols, categorical_cols = identify_feature_types(X)
	preprocessor = build_preprocessor(numeric_cols, categorical_cols)
	preprocessor.fit(X)

	feature_names = preprocessor.get_feature_names_out()
	assert any("categorical__" in name for name in feature_names)


def test_original_dataframe_is_not_modified() -> None:
	original = _sample_frame()
	baseline = original.copy(deep=True)

	X, y = split_features_target(original, "Attrition")
	X = drop_excluded_columns(X, ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])
	_, _ = simulate_missingness(
		X,
		enabled=True,
		fraction=0.1,
		random_seed=42,
		strategy="all_eligible",
	)

	pd.testing.assert_frame_equal(original, baseline)
	assert y.isna().sum() == 0


def test_invalid_input_raises_project_error() -> None:
	with pytest.raises(PreprocessingError, match="DataFrame"):
		load_dataframe(123)  # type: ignore[arg-type]


def test_target_is_not_changed_or_made_missing() -> None:
	df = _sample_frame()
	X, y = split_features_target(df, "Attrition")
	y_before = y.copy(deep=True)

	X = drop_excluded_columns(X, ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])
	_, _ = simulate_missingness(
		X,
		enabled=True,
		fraction=0.1,
		random_seed=11,
		strategy="all_eligible",
	)

	pd.testing.assert_series_equal(y, y_before)
	assert y.isna().sum() == 0


def test_excluded_columns_are_removed() -> None:
	df = _sample_frame()
	X, _ = split_features_target(df, "Attrition")
	result = drop_excluded_columns(X, ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])

	for column in ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]:
		assert column not in result.columns


def test_missingness_simulation_is_deterministic_for_same_seed() -> None:
	df = _sample_frame()
	X, _ = split_features_target(df, "Attrition")
	X = drop_excluded_columns(X, ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])

	X1, meta1 = simulate_missingness(
		X,
		enabled=True,
		fraction=0.2,
		random_seed=7,
		strategy="all_eligible",
	)
	X2, meta2 = simulate_missingness(
		X,
		enabled=True,
		fraction=0.2,
		random_seed=7,
		strategy="all_eligible",
	)

	pd.testing.assert_frame_equal(X1, X2)
	assert meta1 == meta2


def test_unknown_categories_transform_safely_after_fitting() -> None:
	train_df = _sample_frame()
	test_df = _sample_frame()
	test_df.loc[0, "Department"] = "NewDept"

	X_train, _ = split_features_target(train_df, "Attrition")
	X_test, _ = split_features_target(test_df, "Attrition")
	excluded = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
	X_train = drop_excluded_columns(X_train, excluded)
	X_test = drop_excluded_columns(X_test, excluded)

	numeric_cols, categorical_cols = identify_feature_types(X_train)
	preprocessor = build_preprocessor(numeric_cols, categorical_cols)

	_ = transform_features(X_train, preprocessor, fit=True)
	transformed_test = transform_features(X_test, preprocessor, fit=False)
	assert transformed_test.shape[0] == len(X_test)
