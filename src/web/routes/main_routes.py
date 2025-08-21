"""Main web routes for the application."""

from flask import Blueprint, render_template, request
from werkzeug.utils import redirect

from configs import config
from core.exceptions import APIError
from core.validation import validate_dataset
from database.repository import DatabaseRepository
from services.data_service import DataService
from services.prediction_service import PredictionService
from utils.web_utils import (
    get_candlesticks,
    get_dates,
    get_json_for_fig_scatter,
    get_json_for_line_fig,
    get_json_for_line_scatter,
    get_table,
)

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    """Redirect to default dataset."""
    return redirect("Dataset/requirements")


@main_bp.route("/Dataset/<dataset>", methods=["GET"])
def dataset_view(dataset):
    """Display dataset with visualizations."""
    try:
        validate_dataset(dataset)

        db_repo = DatabaseRepository(config.DATASETS_MAPPING[dataset])
        data_service = DataService(db_repo)

        start_date, end_date = get_dates(request.args)
        df = data_service.get_dataset_with_filters(dataset, start_date, end_date)

        if df.empty:
            raise APIError("No data found for the specified date range", 404)

        df_display, heatmap = data_service.prepare_dataset_for_display(df, dataset)

        return render_template(
            "home.jinja",
            title=f"Train Data For {dataset} Dataset For The Past 7 Days",
            df=df_display,
            get_json=get_json_for_line_fig,
            candlestick=get_candlesticks(df_display.SMP) if "SMP" in df_display.columns else None,
            dataset=dataset,
            heatmap=heatmap,
            start_date=start_date,
            end_date=end_date,
        )
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Error displaying dataset: {str(e)}", 500)


@main_bp.route("/Correlation/<dataset>", methods=["GET"])
def correlation_view(dataset):
    """Display correlation analysis."""
    try:
        validate_dataset(dataset)

        db_repo = DatabaseRepository(config.DATASETS_MAPPING[dataset])
        data_service = DataService(db_repo)

        start_date, end_date = get_dates(request.args)
        df = data_service.get_correlation_data(dataset, start_date, end_date)

        if df.empty:
            raise APIError("No data found for correlation analysis", 404)

        return render_template(
            "correlation.jinja",
            title=f"Correlation For {dataset} Dataset For The Past 7 Days",
            df=df,
            get_json=get_json_for_fig_scatter,
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
        )
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Error displaying correlation: {str(e)}", 500)


@main_bp.route("/<model_name>/<dataset>", methods=["GET"])
def model_view(dataset, model_name):
    """Display model predictions and performance."""
    try:
        validate_dataset(dataset)

        db_repo = DatabaseRepository(config.DATASETS_MAPPING[dataset])
        prediction_service = PredictionService(db_repo)

        start_date, end_date = get_dates(request.args)
        df, metrics, history = prediction_service.get_model_predictions(dataset, model_name, start_date, end_date)

        if df.empty:
            raise APIError(f"No predictions found for model {model_name}", 404)

        if "Lstm" in model_name and history is not None:
            best_epoch = None
            if not metrics.empty and "best_epoch" in metrics.columns:
                best_epoch = metrics.iloc[0]["best_epoch"]

            return render_template(
                "lstm.jinja",
                title=f"Model: {model_name}, Dataset: {dataset}, Last 7days Prediction vs Actual Price And Inference",
                chart_json=get_json_for_line_scatter(df, df.columns),
                table=get_table(metrics),
                hist_json=get_json_for_line_scatter(history, history.columns, best_epoch),
                dataset=dataset,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            return render_template(
                "model.jinja",
                title=f"Model: {model_name}, Dataset: {dataset}, Last 7days Prediction vs Actual Price And Inference",
                chart_json=get_json_for_line_scatter(df, df.columns),
                table=get_table(metrics),
                dataset=dataset,
                start_date=start_date,
                end_date=end_date,
            )
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Error displaying model results: {str(e)}", 500)
