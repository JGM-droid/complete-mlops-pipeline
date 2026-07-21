"""Project-specific exception hierarchy for the MLOps pipeline."""


class MLOpsPipelineError(Exception):
	"""Base exception for all project-owned runtime errors."""


class ConfigError(MLOpsPipelineError):
	"""Raised when configuration loading or validation fails."""


class DataValidationError(MLOpsPipelineError):
	"""Raised when dataset schema or quality checks fail."""


class PreprocessingError(MLOpsPipelineError):
	"""Raised when preprocessing setup or transformation fails."""
