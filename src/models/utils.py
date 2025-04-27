from datetime import date

import pandas as pd
from numpy import sqrt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

today = pd.to_datetime(date.today())  # + timedelta(days= 1)


def rmse(y_train, y_train_pred):
    return sqrt(mean_squared_error(y_train, y_train_pred))


def get_metrics_df(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred):
    metrics = pd.DataFrame()
    metrics = metrics.append(
        pd.DataFrame(
            {
                "Train": mean_absolute_error(y_train, y_train_pred),
                "Validation": mean_absolute_error(y_val, y_val_pred),
                "Test": mean_absolute_error(y_test, y_test_pred),
            },
            index=["MAE"],
        )
    )

    metrics = metrics.append(
        pd.DataFrame(
            {
                "Train": rmse(y_train, y_train_pred),
                "Validation": rmse(y_val, y_val_pred),
                "Test": rmse(y_test, y_test_pred),
            },
            index=["RMSE"],
        )
    )

    metrics = metrics.append(
        pd.DataFrame(
            {
                "Train": r2_score(y_train, y_train_pred),
                "Validation": r2_score(y_val, y_val_pred),
                "Test": r2_score(y_test, y_test_pred),
            },
            index=["R2"],
        )
    )

    # We have values =0 So we cant use MAPE
    # scaler = Normalizer()
    # y_train,y_train_pred = scaler.fit_transform([y_train,y_train_pred])
    # y_val,y_val_pred = scaler.fit_transform([y_val,y_val_pred])
    # y_test,y_test_pred = scaler.fit_transform([y_test,y_test_pred])
    # metrics = metrics.append(pd.DataFrame({
    # 	"Train" : mean_absolute_percentage_error(y_train,y_train_pred),
    # 	"Validation" : mean_absolute_percentage_error(y_val,y_val_pred),
    # 	"Test" : mean_absolute_percentage_error(y_test,y_test_pred),
    # },index=['NMAPE']))

    return metrics
