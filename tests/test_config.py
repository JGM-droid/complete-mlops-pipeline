"""Tests for configuration loading and validation behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mlops_pipeline.config import load_config
from mlops_pipeline.exceptions import ConfigError


def test_load_valid_configuration(config_path: Path) -> None:
	config = load_config(config_path)
	assert config["data"]["target_column"] == "Attrition"
	assert config["features"]["missingness_simulation"]["enabled"] is True


def test_load_configuration_missing_file_raises() -> None:
	with pytest.raises(ConfigError, match="Configuration file not found"):
		load_config("configs/does_not_exist.yaml")


def test_load_configuration_malformed_yaml_raises(tmp_path: Path) -> None:
	bad_file = tmp_path / "bad.yaml"
	bad_file.write_text("project: [invalid", encoding="utf-8")

	with pytest.raises(ConfigError, match="Malformed YAML"):
		load_config(bad_file)


def test_load_configuration_missing_required_section_raises(
	config_path: Path, tmp_path: Path
) -> None:
	with config_path.open("r", encoding="utf-8") as handle:
		config = yaml.safe_load(handle)
	config.pop("features")

	invalid_file = tmp_path / "missing_section.yaml"
	invalid_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

	with pytest.raises(ConfigError, match="missing required top-level sections"):
		load_config(invalid_file)


def test_load_configuration_invalid_dataset_path_raises(
	config_path: Path, tmp_path: Path
) -> None:
	with config_path.open("r", encoding="utf-8") as handle:
		config = yaml.safe_load(handle)

	config["data"]["raw_path"] = "data/raw/not_the_verified_file.csv"
	invalid_file = tmp_path / "invalid_dataset_path.yaml"
	invalid_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

	with pytest.raises(ConfigError, match="must reference an existing dataset file"):
		load_config(invalid_file)


def test_load_configuration_invalid_value_type_raises(
	config_path: Path, tmp_path: Path
) -> None:
	with config_path.open("r", encoding="utf-8") as handle:
		config = yaml.safe_load(handle)

	config["project"]["random_seed"] = "forty-two"
	invalid_file = tmp_path / "invalid_types.yaml"
	invalid_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

	with pytest.raises(ConfigError, match="project.random_seed"):
		load_config(invalid_file)
