"""Prediction service for handling ML model operations."""

import logging
from typing import Any

import pandas as pd

from configs import config
from database.repository import DatabaseRepository

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for prediction-related operations."""

    def __init__(self, db_repository: DatabaseRepository):
        self.db = db_repository

    def get_model_predictions(self, dataset_name: str, model_name: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """Get model predictions and metrics."""
        try:
            # Get main prediction data
            where_clause = f'"index" <= "{end_date}" and "index" >= "{start_date}"'
            predictions_df = self.db.get_data("*", model_name, where_clause)

            # Add previous predictions
            previous_predictions = self.db.get_data(f'"index","{model_name}"', "infernce", where_clause)

            if not predictions_df.empty and not previous_predictions.empty:
                predictions_df["Previous Prediction"] = previous_predictions

            # Get model metrics
            metrics = self.db.get_metrics(model_name)

            # Get history for LSTM models
            history = None
            if "Lstm" in model_name:
                try:
                    history = self.db.get_data("*", f"hist_{model_name}")
                except Exception as e:
                    logger.warning(f"Could not get history for {model_name}: {e}")

            return predictions_df, metrics, history

        except Exception as e:
            logger.error(f"Error getting model predictions for {model_name}: {e}")
            return pd.DataFrame(), pd.DataFrame(), None

    def get_all_model_metrics(self, dataset_name: str) -> dict[str, dict]:
        """Get metrics for all models."""
        try:
            all_metrics = {}
            for model in config.MODELS:
                try:
                    metrics = self.db.get_metrics(model)
                    if not metrics.empty:
                        # Select only the relevant columns
                        relevant_cols = ["Train", "Validation", "Test"]
                        available_cols = [col for col in relevant_cols if col in metrics.columns]
                        if available_cols:
                            all_metrics[model] = metrics.loc[:, available_cols].to_dict()
                except Exception as e:
                    logger.warning(f"Could not get metrics for model {model}: {e}")
                    continue

            return all_metrics
        except Exception as e:
            logger.error(f"Error getting all model metrics: {e}")
            return {}

    def get_single_model_metrics(self, model_name: str) -> dict[str, Any]:
        """Get metrics for a single model."""
        try:
            metrics = self.db.get_metrics(model_name)
            if not metrics.empty:
                return metrics.to_dict()
            return {}
        except Exception as e:
            logger.error(f"Error getting metrics for model {model_name}: {e}")
            return {}
