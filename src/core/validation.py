"""Validation utilities for the application."""

from datetime import datetime
from typing import Any

from configs import config

from .exceptions import ValidationError


def validate_dataset(dataset_name: str) -> str:
    """Validate dataset name."""
    if dataset_name not in config.DATASETS_MAPPING:
        raise ValidationError(f"Invalid dataset: {dataset_name}. Valid datasets: {list(config.DATASETS_MAPPING.keys())}")
    return dataset_name


def validate_model(model_name: str) -> str:
    """Validate model name."""
    if model_name not in config.MODELS and model_name != "all":
        raise ValidationError(f"Invalid model: {model_name}. Valid models: {config.MODELS}")
    return model_name


def validate_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    """Validate date range."""
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        if start_dt >= end_dt:
            raise ValidationError("Start date must be before end date")

        return start_date, end_date
    except ValueError as e:
        raise ValidationError(f"Invalid date format: {e}")


def validate_request_args(args: dict[str, Any], required_fields: list = None) -> dict[str, Any]:
    """Validate request arguments."""
    if required_fields:
        missing_fields = [field for field in required_fields if field not in args]
        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")

    return args
