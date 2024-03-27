import pandas as pd
from flask import render_template, Flask, jsonify, request
from werkzeug.utils import redirect

from utils.web_utils import (
    get_dates,
    get_heatmap,
    get_json_for_line_fig,
    get_candlesticks,
    get_json_for_fig_scatter,
    get_json_for_line_scatter,
    get_table,
)

app = Flask(
    __name__,
    static_url_path="",
    static_folder="static",
    template_folder="templates",
)
datasets = [
    "requirements",
    "requirements_units",
    "requirements_weather",
    "requirements_units_weather",
]
datasets_dict = {
    "requirements": "requirements",
    "requirements_units": "requirements_units",
    "requirements_weather": "requirements_weather",
    "requirements_units_weather": "requirements_units_weather",
}

models = ["Linear", "KnnModel", "XgbModel", "Lstm", "Hybrid"]
# dataset = 'requirements'
# database = 'requirements_units'


# Redirections
@app.route("/")
def home():
    return redirect("Dataset/requirements")


@app.route("/Api")
def api_redict():
    return redirect("Api/docs")


# Original data page
@app.route("/Dataset/<dataset>", methods=["GET"])
def index(dataset):
    db = DB(datasets_dict[dataset])
    start_date, end_date = get_dates(request.args)
    df = db.get_data(
        "*", dataset, f'"index" < "{end_date}"and "index" > "{start_date}"'
    )

    if "units" in dataset:
        heatmap = get_heatmap(
            df.iloc[:, 7 : -7 if "cloudCover" in df.columns else -1]
        )
        df = df.drop(
            axis=1,
            columns=df.iloc[:, 6 : -7 if "cloudCover" in df.columns else -1],
        )
    else:
        heatmap = None
    return render_template(
        "home.jinja",
        title=f"Train Data For {dataset} Dataset For The Past 7 Days",
        df=df,
        get_json=get_json_for_line_fig,
        candlestick=get_candlesticks(df.SMP),
        dataset=dataset,
        heatmap=heatmap,
        start_date=start_date,
        end_date=end_date,
    )


# Correlation page
@app.route("/Correlation/<dataset>", methods=["GET"])
def corrolations(dataset):
    db = DB(datasets_dict[dataset])
    start_date, end_date = get_dates(request.args)
    df = db.get_data(
        "*", dataset, f'"index" <= "{end_date}"and "index" >= "{start_date}"'
    )
    if "units" in dataset:
        df = df.drop(
            axis=1,
            columns=df.iloc[:, 6 : -7 if "cloudCover" in df.columns else -1],
        )
    df = df.set_index("SMP").dropna()
    return render_template(
        "correlation.jinja",
        title=f"Correlation For {dataset} Dataset For The Past 7 Days",
        df=df,
        get_json=get_json_for_fig_scatter,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
    )


# Model page
@app.route("/<name>/<dataset>", methods=["GET"])
def page_for_ml_model(dataset, name):
    start_date, end_date = get_dates(request.args)

    db = DB(datasets_dict[dataset])
    df = db.get_data(
        "*", name, f'"index" <= "{end_date}" and "index" >= "{start_date}"'
    )
    df["Previous Prediction"] = db.get_data(
        f'"index","{name}"',
        "infernce",
        f'"index" <= "{end_date}" and "index" >= "{start_date}"',
    )
    metrics = db.get_metrics(name)

    if "Lstm" in name:
        hist = db.get_data("*", f"hist_{name}")
        return render_template(
            "lstm.jinja",
            title=f"Model: {name}, Dataset: {dataset},  Last 7days Prediction vs Actual Price And Inference",
            chart_json=get_json_for_line_scatter(df, df.columns),
            table=get_table(metrics),
            hist_json=get_json_for_line_scatter(
                hist, hist.columns, metrics.iloc[0]["best_epoch"]
            ),
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        return render_template(
            "model.jinja",
            title=f"Model: {name}, Dataset: {dataset},  Last 7days Prediction vs Actual Price And Inference",
            chart_json=get_json_for_line_scatter(df, df.columns),
            table=get_table(metrics),
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
        )


# Api endpoints
@app.route("/Api/<route>")
def api(route):
    if route == "datasets":
        return jsonify(pd.DataFrame(datasets)[0].to_dict())
    elif route == "models":
        return jsonify(pd.DataFrame(models)[0].to_dict())
    elif route == "docs":
        return render_template("api.jinja", datasets=datasets, models=models)


@app.route("/current_prediction/<dataset>")
def current_prediction(dataset):
    try:
        db = DB(datasets_dict[dataset])
        df = pd.DataFrame()
        for model in models:
            df[model] = db.get_data('"index","Inference"', model).dropna()
        df.index = df.index.astype(str)
        return jsonify(df.to_dict())
    except Exception as e:
        print(e)
        return "No Prediction Possible"


@app.route("/previous_prediction/<dataset>", methods=["GET"])
def previous_prediction(dataset):
    try:
        db = DB(datasets_dict[dataset])
        df = pd.DataFrame()
        start_date, end_date = get_dates(request.args)

        for model in models:
            df[model] = db.get_data(
                f'"index","{model}"',
                "infernce",
                f'"index" <= "{end_date}" and "index" >= "{start_date}"',
            )
        df.index = df.index.astype(str)
        return jsonify(df.dropna().to_dict())
    except Exception as e:
        print(e)
        return "No Prediction Possible"


@app.route("/metrics_api/<dataset>/<model>")
def metrics_api(dataset, model):
    db = DB(datasets_dict[dataset])
    try:
        if model == "all":
            dict = {}
            for model in models:
                dict[model] = (
                    db.get_metrics(model)
                    .loc[:, ["Train", "Validation", "Test"]]
                    .to_dict()
                )
            return jsonify(dict)
        else:
            return jsonify(db.get_metrics(model).to_dict())
    except Exception as e:
        print(e)
        return "WRONG"


if __name__ == "__main__":
    app.run(host="localhost", port=434343, debug=True)
