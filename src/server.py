"""
Refactored model training server with improved architecture.
"""

import logging
from pathlib import Path
from threading import Thread

import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sqlmodel import Session, SQLModel, create_engine, func, select
from xgboost import XGBRegressor

from configs import config
from database.repository import DatabaseRepository
from models.lstm.Lstm_model import LSTM
from models.lstm.LstmMVInput import LstmMVInput
from models.model import get_model_results
from models.utils import get_metrics_df
from orm.smp import Dam
from utils.utils import MAE

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", filename=config.LOG_FILE)
logger = logging.getLogger(__name__)


def train_model(model, model_name, df, dataset_name, params):
    """Train a machine learning model and save results."""
    try:
        prediction, metrics = get_model_results(df, params, model_name, model)
        db_out = DatabaseRepository(dataset_name)
        db_out.save_df_to_db(prediction, model_name)
        db_out.save_metrics(metrics, model_name)
        logger.info(f"Successfully trained and saved model {model_name}")
    except Exception as e:
        logger.error(f"Error training model {model_name}: {e}")
        raise


def train_lstm_model(lstm_class, name, df, dataset_name, params):
    """Train LSTM model and save results."""
    try:
        lstm = LstmMVInput(MAE, df, name=f"{name} {dataset_name}", LSTM=lstm_class, **params)
        lstm.train()
        prediction, metrics, hist, best_epoch = lstm.get_results()

        db_out = DatabaseRepository(dataset_name)
        db_out.save_df_to_db(hist, f"hist_{name}")
        db_out.save_df_to_db(prediction, name)

        metrics["best_epoch"] = best_epoch
        db_out.save_metrics(metrics, name)
        logger.info(f"Successfully trained and saved LSTM model {name}")
    except Exception as e:
        logger.error(f"Error training LSTM model {name}: {e}")
        raise


def save_inference(dataset_name):
    """Save inference results for all models."""
    try:
        db = DatabaseRepository(dataset_name)
        inference_df = pd.DataFrame()

        # Collect inference data from all models
        for model in config.MODELS:
            try:
                model_inference = db.get_data('"index","Inference"', model).dropna()
                if not model_inference.empty:
                    inference_df[model] = model_inference
            except Exception as e:
                logger.warning(f"Could not get inference for model {model}: {e}")
                continue

        if not inference_df.empty:
            try:
                # Merge with existing inference data
                existing_inference = db.get_data("*", "infernce")
                if not existing_inference.empty:
                    inference_df = pd.concat([existing_inference, inference_df])
                    inference_df = inference_df.reset_index().drop_duplicates(subset="index").set_index("index")
            except Exception as e:
                logger.warning(f"Could not merge with existing inference data: {e}")

            db.save_df_to_db(inference_df, "infernce")
            logger.info(f"Successfully saved inference data for dataset {dataset_name}")
        else:
            logger.warning(f"No inference data available for dataset {dataset_name}")

    except Exception as e:
        logger.error(f"Error saving inference for dataset {dataset_name}: {e}")
        raise


# Model configurations
MODEL_CLASSES = [LinearRegression, KNeighborsRegressor, XGBRegressor, LSTM]
MODEL_NAMES = ["Linear", "KnnModel", "XgbModel"]
PARAM_NAMES = ["linear_params", "knn_params", "xgb_params"]


def load_model_parameters():
    """Load model parameters from YAML configuration."""
    try:
        yaml_path = Path("../yaml.yaml")
        if yaml_path.exists():
            with open(yaml_path) as file:
                return yaml.safe_load(file)
        else:
            logger.warning("Model parameters YAML file not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading model parameters: {e}")
        return {}


def initialize_database():
    """Initialize the database and check for existing data."""
    try:
        engine = create_engine(config.DATABASE_URL)
        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            stmt = select(func.max(Dam.timestamp))
            max_date = session.exec(stmt).first()

        if max_date is None:
            max_date = config.START_DATE
            logger.info("Database initialized with start date")
        else:
            logger.info(f"Database already contains data up to: {max_date}")

        return max_date
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def prepare_dataset(db_repo: DatabaseRepository, dataset_name: str) -> pd.DataFrame:
    """Prepare dataset for training by adding lag features."""
    try:
        dataset = db_repo.get_dataset("*", dataset_name)
        if dataset.empty:
            raise ValueError(f"No data found for dataset {dataset_name}")

        # Add lag features
        dataset.insert(dataset.shape[1] - 1, "lag_24", dataset["SMP"].shift(config.INFERENCE_PERIOD_HOURS))
        dataset = dataset[dataset["lag_24"].notna()]
        dataset = dataset[~dataset.index.duplicated(keep="first")]

        logger.info(f"Prepared dataset {dataset_name} with {len(dataset)} records")
        return dataset
    except Exception as e:
        logger.error(f"Error preparing dataset {dataset_name}: {e}")
        raise


def train_all_models():
    """Train all models for all datasets."""
    try:
        params_list = load_model_parameters()
        if not params_list:
            logger.error("No model parameters loaded, cannot train models")
            return

        db_in = DatabaseRepository("dataset")
        threads = []

        for dataset_name in config.DATASETS:
            logger.info(f"Processing dataset: {dataset_name}")

            # Save inference data first
            save_inference(dataset_name)

            # Prepare dataset
            dataset = prepare_dataset(db_in, dataset_name)

            # Train standard ML models
            for model_class, model_name, param_name in zip(MODEL_CLASSES[:-1], MODEL_NAMES, PARAM_NAMES, strict=False):
                if param_name in params_list:
                    thread = Thread(target=train_model, args=(model_class, model_name, dataset, dataset_name, params_list[param_name]))
                    threads.append(thread)
                    thread.start()
                else:
                    logger.warning(f"Parameters not found for {param_name}")

            # Train LSTM model
            if "Lstm_params" in params_list:
                lstm_thread = Thread(target=train_lstm_model, args=(LSTM, "Lstm", dataset, dataset_name, params_list["Lstm_params"]))
                threads.append(lstm_thread)
                lstm_thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        logger.info("All model training completed")

        # Train hybrid models
        train_hybrid_models()

    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise


def train_hybrid_models():
    """Train hybrid ensemble models."""
    try:
        model_names_with_lstm = MODEL_NAMES + ["Lstm"]

        for dataset_name in config.DATASETS:
            logger.info(f"Training hybrid model for dataset: {dataset_name}")

            db = DatabaseRepository(dataset_name)
            model_data = {}

            # Collect data from all individual models
            for model_name in model_names_with_lstm:
                try:
                    model_data[model_name] = db.get_data("*", model_name)
                except Exception as e:
                    logger.warning(f"Could not load data for model {model_name}: {e}")
                    continue

            if not model_data:
                logger.warning(f"No model data available for hybrid training on {dataset_name}")
                continue

            # Get base SMP data
            smp_data = None
            for model_name, data in model_data.items():
                if not data.empty and "SMP" in data.columns:
                    smp_data = data["SMP"]
                    break

            if smp_data is None:
                logger.warning(f"No SMP data found for hybrid model on {dataset_name}")
                continue

            # Create ensemble predictions
            ensemble_data = {}
            for phase in ["Training", "Validation", "Testing", "Inference"]:
                phase_predictions = []
                for model_name, data in model_data.items():
                    if not data.empty and phase in data.columns:
                        clean_predictions = data[phase].dropna()
                        if not clean_predictions.empty:
                            phase_predictions.append(clean_predictions)

                if phase_predictions:
                    ensemble_data[phase] = pd.concat(phase_predictions, axis=1).mean(axis=1)

            # Create final prediction DataFrame
            hybrid_prediction = pd.DataFrame({"SMP": smp_data, **ensemble_data})

            # Calculate hybrid model metrics
            if all(phase in ensemble_data for phase in ["Training", "Validation", "Testing"]):
                hybrid_metrics = get_metrics_df(
                    hybrid_prediction.loc[:, ["SMP", "Training"]].dropna()["SMP"],
                    ensemble_data["Training"].dropna(),
                    hybrid_prediction.loc[:, ["SMP", "Validation"]].dropna()["SMP"],
                    ensemble_data["Validation"].dropna(),
                    hybrid_prediction.loc[:, ["SMP", "Testing"]].dropna()["SMP"],
                    ensemble_data["Testing"].dropna(),
                )

                # Save hybrid model results
                db.save_df_to_db(hybrid_prediction, "Hybrid")
                db.save_metrics(hybrid_metrics, "Hybrid")
                logger.info(f"Successfully trained and saved hybrid model for {dataset_name}")
            else:
                logger.warning(f"Insufficient data for hybrid model metrics on {dataset_name}")

    except Exception as e:
        logger.error(f"Error training hybrid models: {e}")
        raise


def main():
    """Main training pipeline."""
    try:
        logger.info("Starting model training pipeline")

        # Initialize database
        initialize_database()

        # Train all models
        train_all_models()

        logger.info("Model training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Model training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
