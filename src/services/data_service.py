"""Data service for handling data operations."""

import logging
from typing import Any

import pandas as pd

from configs import config
from database.repository import DatabaseRepository

logger = logging.getLogger(__name__)


class DataService:
    """Service for data-related operations."""

    def __init__(self, db_repository: DatabaseRepository):
        self.db = db_repository

    def get_dataset_with_filters(self, dataset_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get dataset with date filters applied."""
        try:
            where_clause = f'"index" < "{end_date}" and "index" > "{start_date}"'
            return self.db.get_data("*", dataset_name, where_clause)
        except Exception as e:
            logger.error(f"Error getting dataset {dataset_name}: {e}")
            return pd.DataFrame()

    def prepare_dataset_for_display(self, df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, str | None]:
        """Prepare dataset for display, including heatmap generation."""
        heatmap = None

        if "units" in dataset_name and not df.empty:
            # Create heatmap for units datasets
            heatmap_cols = df.iloc[:, 7 : -7 if "cloudCover" in df.columns else -1]
            if not heatmap_cols.empty:
                from utils.web_utils import get_heatmap

                heatmap = get_heatmap(heatmap_cols)

            # Remove unit columns for display
            df = df.drop(
                axis=1,
                columns=df.iloc[:, 6 : -7 if "cloudCover" in df.columns else -1],
            )

        return df, heatmap

    def get_correlation_data(self, dataset_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get data prepared for correlation analysis."""
        where_clause = f'"index" <= "{end_date}" and "index" >= "{start_date}"'
        df = self.db.get_data("*", dataset_name, where_clause)

        if "units" in dataset_name and not df.empty:
            df = df.drop(
                axis=1,
                columns=df.iloc[:, 6 : -7 if "cloudCover" in df.columns else -1],
            )

        if not df.empty:
            df = df.set_index("SMP").dropna()

        return df

    def get_current_predictions(self, dataset_name: str) -> dict[str, Any]:
        """Get current predictions for all models."""
        try:
            predictions = {}
            for model in config.MODELS:
                try:
                    model_data = self.db.get_data('"index","Inference"', model)
                    if not model_data.empty:
                        predictions[model] = model_data.dropna()
                except Exception as e:
                    logger.warning(f"Could not get predictions for model {model}: {e}")
                    continue

            # Convert index to string for JSON serialization
            for model, data in predictions.items():
                data.index = data.index.astype(str)

            return predictions
        except Exception as e:
            logger.error(f"Error getting current predictions: {e}")
            return {}

    def get_previous_predictions(self, dataset_name: str, start_date: str, end_date: str) -> dict[str, Any]:
        """Get historical predictions for all models."""
        try:
            where_clause = f'"index" <= "{end_date}" and "index" >= "{start_date}"'
            predictions = {}

            for model in config.MODELS:
                try:
                    columns = f'"index","{model}"'
                    model_data = self.db.get_data(columns, "infernce", where_clause)
                    if not model_data.empty:
                        predictions[model] = model_data
                except Exception as e:
                    logger.warning(f"Could not get historical predictions for model {model}: {e}")
                    continue

            # Convert index to string for JSON serialization
            for model, data in predictions.items():
                data.index = data.index.astype(str)

            return predictions
        except Exception as e:
            logger.error(f"Error getting previous predictions: {e}")
            return {}
