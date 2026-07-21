"""Preprocessing utilities for Phase 2 tabular workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .exceptions import PreprocessingError


def load_dataframe(data: pd.DataFrame | str | Path) -> pd.DataFrame:
	"""Load a DataFrame from memory or CSV path without mutating caller data."""
	if isinstance(data, pd.DataFrame):
		return data.copy(deep=True)

	if isinstance(data, (str, Path)):
		csv_path = Path(data)
		if not csv_path.exists() or not csv_path.is_file():
			raise PreprocessingError(f"Dataset path is invalid or missing: {csv_path}")
		try:
			return pd.read_csv(csv_path)
		except Exception as exc:
			raise PreprocessingError(f"Failed to read dataset CSV '{csv_path}': {exc}") from exc

	raise PreprocessingError(
		"Input data must be a pandas DataFrame or a CSV path string/pathlib.Path."
	)


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
	"""Split DataFrame into feature matrix X and target vector y."""
	_ensure_dataframe(df)
	if target_column not in df.columns:
		raise PreprocessingError(f"Target column '{target_column}' is missing from dataset.")

	X = df.drop(columns=[target_column]).copy(deep=True)
	y = df[target_column].copy(deep=True)
	return X, y


def drop_excluded_columns(X: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
	"""Drop explicitly excluded feature columns from X."""
	_ensure_dataframe(X)
	if not isinstance(excluded_columns, list) or not all(
		isinstance(col, str) for col in excluded_columns
	):
		raise PreprocessingError("'excluded_columns' must be a list of strings.")

	cols_to_drop = [col for col in excluded_columns if col in X.columns]
	return X.drop(columns=cols_to_drop).copy(deep=True)


def identify_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
	"""Identify numeric and categorical feature columns."""
	_ensure_dataframe(X)
	numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
	categorical_cols = [col for col in X.columns if col not in numeric_cols]

	if not numeric_cols:
		raise PreprocessingError("No numeric feature columns detected.")
	if not categorical_cols:
		raise PreprocessingError("No categorical feature columns detected.")

	return numeric_cols, categorical_cols


def simulate_missingness(
	X: pd.DataFrame,
	*,
	enabled: bool,
	fraction: float,
	random_seed: int,
	strategy: str,
	columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
	"""Apply deterministic feature-only missingness simulation.

	Missing values are introduced in selected feature columns only.
	"""
	_ensure_dataframe(X)

	X_out = X.copy(deep=True)
	metadata: dict[str, Any] = {
		"enabled": enabled,
		"fraction": float(fraction),
		"random_seed": random_seed,
		"strategy": strategy,
		"applied_columns": [],
		"missing_cells_introduced": 0,
	}

	if not enabled:
		return X_out, metadata

	if not isinstance(fraction, (int, float)) or not 0 < float(fraction) <= 0.2:
		raise PreprocessingError("Missingness fraction must be numeric and in the range (0, 0.2].")
	if not isinstance(random_seed, int):
		raise PreprocessingError("Missingness random seed must be an integer.")

	numeric_cols, categorical_cols = identify_feature_types(X_out)
	if strategy == "all_eligible":
		selected_cols = X_out.columns.tolist()
	elif strategy == "selected_columns":
		if not columns:
			raise PreprocessingError(
				"Missingness strategy 'selected_columns' requires a non-empty columns list."
			)
		unknown = [col for col in columns if col not in X_out.columns]
		if unknown:
			raise PreprocessingError(
				f"Missingness simulation columns are not in feature set: {unknown}"
			)
		selected_cols = list(columns)
	else:
		raise PreprocessingError(
			"Missingness strategy must be 'all_eligible' or 'selected_columns'."
		)

	has_numeric = any(col in numeric_cols for col in selected_cols)
	has_categorical = any(col in categorical_cols for col in selected_cols)
	if not (has_numeric and has_categorical):
		raise PreprocessingError(
			"Missingness simulation must affect both numeric and categorical features."
		)

	rng = np.random.RandomState(random_seed)
	rows = len(X_out)
	cells_per_column = max(1, int(np.floor(rows * float(fraction))))

	total_missing = 0
	for column in selected_cols:
		indices = rng.choice(rows, size=cells_per_column, replace=False)
		X_out.loc[X_out.index[indices], column] = np.nan
		total_missing += cells_per_column

	metadata["applied_columns"] = selected_cols
	metadata["missing_cells_introduced"] = int(total_missing)
	return X_out, metadata


def build_preprocessor(
	numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
	"""Build sklearn preprocessing pipeline for numeric and categorical features."""
	if not numeric_columns:
		raise PreprocessingError("Numeric column list cannot be empty.")
	if not categorical_columns:
		raise PreprocessingError("Categorical column list cannot be empty.")

	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
		]
	)

	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			(
				"onehot",
				OneHotEncoder(handle_unknown="ignore", sparse_output=False),
			),
		]
	)

	return ColumnTransformer(
		transformers=[
			("numeric", numeric_pipeline, numeric_columns),
			("categorical", categorical_pipeline, categorical_columns),
		]
	)


def transform_features(
	X: pd.DataFrame,
	preprocessor: ColumnTransformer,
	*,
	fit: bool = False,
) -> np.ndarray:
	"""Transform feature matrix with optional fitting step."""
	_ensure_dataframe(X)
	if not isinstance(preprocessor, ColumnTransformer):
		raise PreprocessingError("preprocessor must be an sklearn ColumnTransformer.")

	if fit:
		return preprocessor.fit_transform(X)
	return preprocessor.transform(X)


def _ensure_dataframe(df: Any) -> None:
	if not isinstance(df, pd.DataFrame):
		raise PreprocessingError(
			f"Expected pandas DataFrame, received {type(df).__name__}."
		)
