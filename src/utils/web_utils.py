import json
from datetime import date, timedelta

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pytz import timezone

template = pio.templates["plotly_dark"]
localTz = timezone("CET")


def get_json_for_line_fig(df, x, y):
    fig = px.line(df, x=x, y=y, template=template)
    fig = fig.update_xaxes(rangeslider_visible=True)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def get_json_for_line_scatter(df, y, line=None):
    fig = go.Figure()
    fig.update_layout(template=template)
    for i in y:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[i], mode="lines+markers", name=df[i].name
            )
        )
        if line is not None:
            fig.add_vline(
                x=line,
                line_dash="dash",
                line_color="purple",
                annotation_text="Best Epoch",
            )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def get_json_for_fig_scatter(df, y, x):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        trendline="ols",
        trendline_color_override="red",
        template=template,
    )
    fig = fig.update_xaxes(rangeslider_visible=True)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def get_candlesticks(df):
    export = pd.DataFrame()
    # export['High'] = [df[i:i+24].max() for i in range(0,len(df),24)]
    for i in range(0, len(df), 24):
        temp = pd.DataFrame()
        temp["Date"] = [df[i : i + 24].index[0]]
        temp["high"] = df[i : i + 24].max()
        temp["low"] = df[i : i + 24].min()
        temp["open"] = df[i : i + 24].iloc[0]
        temp["close"] = df[i : i + 24].iloc[-1]
        temp = temp.set_index("Date")
        export = pd.concat([export, temp])
        candlestick = go.Candlestick(
            x=export.index,
            open=export["open"],
            high=export["high"],
            low=export["low"],
            close=export["close"],
        )

    fig = go.Figure(data=[candlestick])
    fig.update_layout(template=template)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def get_heatmap(df):
    units_24 = df.copy()
    units_24 = pd.DataFrame(
        [units_24.iloc[i] for i in range(0, units_24.shape[0], 24)]
    )
    units_24.index = [
        units_24.index[i].date() for i in range(units_24.shape[0])
    ]
    units_24.head()
    units_24_trans = units_24.iloc[:, :-1].transpose()

    heatmap = go.Heatmap(
        z=units_24_trans.values,
        x=units_24_trans.columns,
        y=units_24_trans.index,
    )
    fig = go.Figure(data=[heatmap])
    fig.update_layout(template=template)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def get_table(df):
    df = df.round(4).reset_index()
    df.columns.values[0] = "Metric"
    return df


def get_dates(form=""):
    if "end_date" in form:
        end_date = form["end_date"]
    else:
        end_date = str(pd.to_datetime(date.today() + timedelta(days=2)))[:10]

    if "start_date" in form:
        start_date = form["start_date"]
    else:
        start_date = str(
            localTz.localize(pd.to_datetime(date.today()) - timedelta(weeks=1))
        )[:10]

    return start_date, end_date
