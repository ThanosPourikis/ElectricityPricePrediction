# from configs import CITIES, DARK_SKY_KEY
import requests
import json
from pytz import timezone
from datetime import datetime, timedelta, date
import pandas as pd

path = "datasets/weather_data/"


def download_weather_data():
    exclude = "exclude=currently,minutely,daily,alerts,flags"
    to_keep = {
        "time": 0,
        "temperature": 0,
        "windSpeed": 0,
        "windGust": 0,
        "cloudCover": 0,
        "uvIndex": 0,
        "visibility": 0,
        "city": "",
    }
    feature_list = [
        "time",
        "temperature",
        "windSpeed",
        "windGust",
        "cloudCover",
        "uvIndex",
        "visibility",
    ]

    end = datetime.date(datetime.now() + timedelta(days=3))
    localTz = timezone("CET")
    df = pd.DataFrame(columns=to_keep)
    time = int(datetime.fromisoformat("2020-11-01 00:00:00").timestamp())

    timedelt = 86400

    for city in CITIES:
        try:
            df = pd.read_csv(f"{path}{city}.csv")
            time = int(datetime.fromisoformat(df.iloc[-1]["time"]).timestamp())
        except Exception as e:
            print(e)
            df = pd.DataFrame(columns=to_keep)
            time = int(
                datetime.fromisoformat("2020-11-01 00:00:00").timestamp()
            )

        lat = CITIES[city][0]
        lon = CITIES[city][1]
        to_keep["city"] = city
        while date.fromtimestamp(time) != end:
            url = f"https://api.darksky.net/forecast/{DARK_SKY_KEY}/{lat},{lon},{time}?units=si&{exclude}"
            response = requests.get(url)
            k = json.loads(response.text)
            for i in k["hourly"]["data"]:
                for j in feature_list:
                    to_keep[j] = i[j]
                to_keep["time"] = localTz.localize(
                    datetime.fromtimestamp(to_keep["time"])
                )
                df = df.append(to_keep, ignore_index=True)
            time += timedelt
        df.set_index("time").to_csv(f"{path}{city}.csv")


def get_weather_data():
    export = (
        pd.DataFrame(pd.read_csv(f"{path}/Athens.csv")["time"])
        .rename(columns={"time": "Date"})
        .set_index("Date")
    )

    for city in CITIES:
        df = pd.read_csv(f"{path}{city}.csv")
        df = df.rename(
            columns={
                "time": "Date",
                "temperature": f"temperature_{city}",
                "windSpeed": f"windSpeed_{city}",
                "windGust": f"windGust_{city}",
                "cloudCover": f"cloudCover_{city}",
                "uvIndex": f"uvIndex_{city}",
                "visibility": f"visibility_{city}",
            }
        )

        df = df.loc[:, df.columns != "city"]
        export = export.join(df.set_index("Date"))
    return export.reset_index()


def get_weather_mean():
    cities = [
        "Hraklio.csv",
        "Larissa.csv",
        "Patra.csv",
        "Pireas.csv",
        "Thessaloniki.csv",
    ]
    weather = pd.read_csv(f"{path}Athens.csv").iloc[:, :-1].set_index("time")
    for i in cities:
        df = pd.read_csv(f"{path}{i}").iloc[:, :-1].set_index("time")
    weather = weather + df

    weather = weather / (len(cities) + 1)
    weather["Date"] = weather.index
    weather = weather.set_index("Date")
    weather.index = pd.to_datetime(weather.index)
    return weather
