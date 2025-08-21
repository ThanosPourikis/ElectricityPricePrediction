"""Configuration settings for the electricity price prediction application."""

import os
from datetime import datetime
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
FOLDER_PATH = os.getenv("FOLDER_PATH", "/tmp/files")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/electricity_prediction.db")

# API Configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", 5000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Model Configuration
VALIDATION_SIZE = float(os.getenv("VALIDATION_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 96))

# Time periods
INFERENCE_PERIOD_HOURS = 24
TEST_PERIOD_DAYS = 8
DEFAULT_LOOKBACK_DAYS = 7

# Date configuration
START_DATE = datetime(2020, 1, 1)

# Dataset names
DATASETS = [
    "requirements",
    "requirements_units",
    "requirements_weather",
    "requirements_units_weather",
]

DATASETS_MAPPING = {name: name for name in DATASETS}

# Model names
MODELS = ["Linear", "KnnModel", "XgbModel", "Lstm", "Hybrid"]

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
