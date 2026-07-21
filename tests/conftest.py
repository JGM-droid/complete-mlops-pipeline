"""Shared deterministic fixtures for Phase 2 tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
	return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path(repo_root: Path) -> Path:
	return repo_root / "configs" / "train.yaml"


@pytest.fixture(scope="session")
def dataset_path(repo_root: Path) -> Path:
	return repo_root / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


@pytest.fixture(scope="session")
def raw_dataset_df(dataset_path: Path) -> pd.DataFrame:
	return pd.read_csv(dataset_path)


@pytest.fixture(scope="session")
def target_column() -> str:
	return "Attrition"


@pytest.fixture(scope="session")
def excluded_columns() -> list[str]:
	return ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]


@pytest.fixture(scope="session")
def required_columns() -> list[str]:
	return [
		"Age",
		"Attrition",
		"DistanceFromHome",
		"EmployeeCount",
		"EmployeeNumber",
		"Over18",
		"PercentSalaryHike",
		"StandardHours",
		"TotalWorkingYears",
		"YearsAtCompany",
	]
