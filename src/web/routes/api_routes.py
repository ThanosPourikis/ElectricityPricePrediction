"""API routes for the application."""

import pandas as pd
from flask import Blueprint, jsonify, redirect, render_template, request

from configs import config
from core.exceptions import APIError
from core.validation import validate_dataset, validate_model
from database.repository import DatabaseRepository
from services.data_service import DataService
from services.prediction_service import PredictionService
from utils.web_utils import get_dates

api_bp = Blueprint("api", __name__)


@api_bp.route("/")
def api_redirect():
    """Redirect to API documentation."""
    return redirect("docs")


@api_bp.route("/docs")
def api_docs():
    """API documentation page."""
    return render_template("api.jinja", datasets=config.DATASETS, models=config.MODELS)


@api_bp.route("/datasets")
def get_datasets():
    """Get available datasets."""
    try:
        return jsonify(pd.DataFrame(config.DATASETS)[0].to_dict())
    except Exception as e:
        raise APIError(f"Error retrieving datasets: {str(e)}", 500)


@api_bp.route("/models")
def get_models():
    """Get available models."""
    try:
        return jsonify(pd.DataFrame(config.MODELS)[0].to_dict())
    except Exception as e:
        raise APIError(f"Error retrieving models: {str(e)}", 500)


@api_bp.route("/current_prediction/<dataset>")
def current_prediction(dataset):
    """Get current predictions for all models."""
    try:
        validate_dataset(dataset)

        db_repo = DatabaseRepository(config.DATASETS_MAPPING[dataset])
        data_service = DataService(db_repo)

        predictions = data_service.get_current_predictions(dataset)

        if not predictions:
            raise APIError("No current predictions available", 404)

        # Convert DataFrames to dictionaries for JSON serialization
        json_predictions = {}
        for model, df in predictions.items():
            if not df.empty:
                json_predictions[model] = df.to_dict()

        return jsonify(json_predictions)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Error getting current predictions: {str(e)}", 500)


@api_bp.route("/previous_prediction/<dataset>", methods=["GET"])
def previous_prediction(dataset):
    """Get historical predictions for all models."""
    try:
        validate_dataset(dataset)

        db_repo = DatabaseRepository(config.DATASETS_MAPPING[dataset])
        data_service = DataService(db_repo)

        start_date, end_date = get_dates(request.args)
        predictions = data_service.get_previous_predictions(dataset, start_date, end_date)

        if not predictions:
            raise APIError("No historical predictions found", 404)

        # Convert DataFrames to dictionaries and filter out empty ones
        json_predictions = {}
        for model, df in predictions.items():
            if not df.empty:
                json_predictions[model] = df.to_dict()

        return jsonify(json_predictions)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Error getting previous predictions: {str(e)}", 500)


@api_bp.route("/metrics/<dataset>/<model>")
def metrics_api(dataset, model):
    """Get model metrics."""
    try:
        validate_dataset(dataset)
        validate_model(model)

        db_repo = DatabaseRepository(config.DATASETS_MAPPING[dataset])
        prediction_service = PredictionService(db_repo)

        if model == "all":
            metrics = prediction_service.get_all_model_metrics(dataset)
        else:
            metrics = prediction_service.get_single_model_metrics(model)

        if not metrics:
            raise APIError(f"No metrics found for model(s): {model}", 404)

        return jsonify(metrics)
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Error getting model metrics: {str(e)}", 500)
