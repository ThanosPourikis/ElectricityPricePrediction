import datetime
import logging
import sys
from logging.handlers import RotatingFileHandler

import pandas as pd
from data.get_weather_data import get_weather_data
from dotenv import load_dotenv
from os import getenv

load_dotenv()
MSE = "MSE"
MAE = "MAE"
HuberLoss = "HuberLoss"


def get_req():
    return (
        pd.read_csv("datasets/requirements.csv")
        .set_index("Date")
        .join(pd.read_csv("datasets/SMP.csv").set_index("Date"))
        .reset_index()
    )


def get_data_from_csv():
    return (
        pd.read_csv("datasets/requirements.csv")
        .set_index("Date")
        .join(pd.read_csv("datasets/units.csv").set_index("Date"))
        .join(pd.read_csv("datasets/SMP.csv").set_index("Date"))
        .reset_index()
    )


def get_data_from_csv_with_weather():
    return (
        get_data_from_csv()
        .set_index("Date")
        .join(get_weather_data().set_index("Date"))
    ).reset_index()


def date_con(date):
    return datetime.datetime.fromisoformat(date)


def check_data_integrity(df):
    print(len(df) / 24)
    df.to_csv("check_set")
    df = df["Date"]
    delt = datetime.timedelta(hours=1)
    for i in range(len(df) - 1):
        if date_con(df[i]) + delt != date_con(df[i + 1]):
            print(f"Error at {df[i]}")


def init_logger():
    class LessThanErrorFilter(logging.Filter):
        def filter(self, record):
            return record.levelno < logging.ERROR

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = RotatingFileHandler(
        getenv("LOG_FILE"), maxBytes=5 * 1024 * 1024, backupCount=2
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(LessThanErrorFilter())
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)


init_logger()
