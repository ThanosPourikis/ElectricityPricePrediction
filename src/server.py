import logging
import os
from datetime import datetime
from threading import Thread

import pandas as pd
import yaml
from sklearn import utils
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sqlmodel import Session, create_engine, select, func, SQLModel
from xgboost import XGBRegressor

# from data.get_SMP_data import rebuild_db
from models.lstm.LstmMVInput import LstmMVInput
from models.lstm.Lstm_model import LSTM
from models.model import get_model_results
from models.utils import get_metrics_df
from orm.smp import Dam
from utils import utils
from utils.update_data import update_data
from configs import config


def train_model(model, model_name, df, dataset_name, params):
    prediction, metrics = get_model_results(
        df, params[dataset_name], model_name, model
    )
    db_out = DB(dataset_name)
    db_out.save_df_to_db(prediction, model_name)
    db_out.save_metrics(metrics, model_name)


def Lstm(LSTM, name, df, dataset_name, params):
    lstm = LstmMVInput(
        utils.MAE, df, name=f"{name} {dataset_name}", LSTM=LSTM, **params
    )
    lstm.train()
    prediction, metrics, hist, best_epoch = lstm.get_results()
    db_out = DB(dataset_name)

    db_out.save_df_to_db(hist, f"hist_{name}")
    db_out.save_df_to_db(prediction, name)
    metrics["best_epoch"] = best_epoch
    db_out.save_metrics(metrics, name)


def save_infernce(dataset_name):
    try:
        db = DB(dataset_name)
        df = pd.DataFrame()
        df["Linear"] = db.get_data('"index","Inference"', "Linear").dropna()
        df["KnnModel"] = db.get_data(
            '"index","Inference"', "KnnModel"
        ).dropna()
        df["XgbModel"] = db.get_data(
            '"index","Inference"', "XgbModel"
        ).dropna()
        df["Lstm"] = db.get_data('"index","Inference"', "Lstm").dropna()
        df["Hybrid"] = db.get_data('"index","Inference"', "Hybrid").dropna()
        try:
            df = pd.concat([db.get_data("*", "infernce"), df])
            df = (
                df.reset_index()
                .drop_duplicates(subset="index")
                .set_index("index")
            )
        except Exception as e:
            print(e)
            pass
        db.save_df_to_db(df, "infernce")
    except Exception as e:
        print(e)
        return "No Prediction Possible"


logging.basicConfig(filename="log.log", level=logging.DEBUG)

datasets = [
    "requirements",
    "requirements_units",
    "requirements_weather",
    "requirements_units_weather",
]
database_in = "dataset"
models = [LinearRegression, KNeighborsRegressor, XGBRegressor, LSTM]
model_names = ["Linear", "KnnModel", "XgbModel"]
params = ["linear_params", "knn_params", "xgb_params"]

try:
    with open("../yaml.yaml", "r") as file:
        params_list = yaml.safe_load(file)
except Exception as e:
    print(e)


def main():
    engine = create_engine(os.getenv("DATABASE_URL"))
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        stm = select(func.max(Dam.timestamp))
        max_date = session.exec(stm).first()
    if max_date is None:
        max_date = config.START_DATE
    # rebuild_db(config.FOLDER_PATH)


def to_refactor():
    db_in = DB(database_in)
    threads = []
    for dataset_name in datasets:
        save_infernce(dataset_name)
        dataset = db_in.get_dataset("*", dataset_name)
        dataset.insert(
            dataset.shape[1] - 1, "lag_24", dataset["SMP"].shift(24)
        )
        dataset = dataset[dataset["lag_24"].notna()]
        dataset = dataset[~dataset.index.duplicated(keep="first")]

        for model, model_name, param in zip(models, model_names, params):
            Thread(
                target=train_model,
                args=(
                    model,
                    model_name,
                    dataset,
                    dataset_name,
                    params_list[param],
                ),
            ).start()
        threads.append(
            Thread(
                target=Lstm,
                args=(
                    LSTM,
                    "Lstm",
                    dataset,
                    dataset_name,
                    params_list["Lstm_params"],
                ),
            )
        )

    for i in threads:
        i.start()

    for i in threads:
        i.join()

    model_names.append("LSTM")

    for dataset in datasets:
        db = DB(dataset)
        data = {}
        for i in model_names:
            data[i] = db.get_data("*", i)

        smp = data["Linear"]["SMP"]

        val = pd.concat(
            [data[i]["Validation"].dropna() for i in data], axis=1
        ).mean(axis=1)

        test = pd.concat(
            [data[i]["Testing"].dropna() for i in data], axis=1
        ).mean(axis=1)

        inf = pd.concat(
            [data[i]["Inference"].dropna() for i in data], axis=1
        ).mean(axis=1)

        data.pop("KnnModel", None)
        train = pd.concat(
            [data[i]["Training"].dropna() for i in data], axis=1
        ).mean(axis=1)

        prediction = pd.concat([smp, train, val, test, inf], axis=1)
        prediction.columns = [
            "SMP",
            "Training",
            "Validation",
            "Testing",
            "Inference",
        ]

        metrics = get_metrics_df(
            prediction.loc[:, ["SMP", "Training"]].dropna()["SMP"],
            prediction["Training"].dropna(),
            prediction.loc[:, ["SMP", "Validation"]].dropna()["SMP"],
            prediction["Validation"].dropna(),
            prediction.loc[:, ["SMP", "Testing"]].dropna()["SMP"],
            prediction["Testing"].dropna(),
        )

        prediction, metrics
        db_out = DB(dataset)

        db_out.save_df_to_db(prediction, "Hybrid")
        db_out.save_metrics(metrics, "Hybrid")


if __name__ == "__main__":
    main()
