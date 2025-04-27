import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from models.utils import get_metrics_df


def get_model_results(data, params, name, model, validation_size=0.2):
    # features = data.loc[:,data.columns!='SMP']
    # data = data.set_index('Date')
    export = pd.DataFrame(data["SMP"], data.index)
    if data.isnull().values.any():
        inference = data[-24:]
        test = data[-(8 * 24) : -24]
        data = data[: -(8 * 24)]
    else:
        test = data[-(7 * 24) :]
        data = data[: -(7 * 24)]
    features = data.loc[:, data.columns != "SMP"]
    labels = data.loc[:, "SMP"]
    model = model(**params)

    x_train, x_validate, y_train, y_validate = train_test_split(
        features,
        labels,
        random_state=96,
        test_size=validation_size,
        shuffle=True,
    )

    logging.info(f"Fitting {name} Model")

    model.fit(x_train, y_train)

    train = model.predict(x_train)
    val = model.predict(x_validate)
    pred = model.predict(test.loc[:, test.columns != "SMP"])
    metrics = get_metrics_df(y_train, train, y_validate, val, test.loc[:, "SMP"], pred)

    export["Training"] = pd.DataFrame(train, index=x_train.index)
    export["Validation"] = pd.DataFrame(val, index=x_validate.index)
    export["Testing"] = pd.DataFrame(pred, index=test.index)
    try:
        export["Inference"] = pd.DataFrame(
            model.predict(inference.loc[:, inference.columns != "SMP"]),
            index=inference.index,
        )
    finally:
        return export, metrics
