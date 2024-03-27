from datetime import datetime
from pathlib import Path

import pandas as pd

import configs
from data.ADMHE_files import get_excel_data
from data.get_SMP_data import get_SMP_data
from data.get_weather_data import download_weather_data, get_weather_mean
from data.isp_data import get_isp_data
from data.units_data import get_unit_data


def update_data(folder_path: Path, curr_date: datetime):
    try:
        start_date = db.get_data('MAX("index")', "requirements").values[0, 0]
        requirements = get_isp_data(
            get_excel_data(
                folder_path=configs.ISP1["folder_path"],
                filetype=configs.ISP1["filetype"],
                start_date=start_date,
            )
        )
        requirements = pd.concat(
            [db.get_data("*", "requirements"), requirements]
        )
    except Exception as e:
        print(e)
        start_date = "2020-11-01"
        requirements = get_isp_data(
            get_excel_data(
                folder_path=configs.ISP1["folder_path"],
                filetype=configs.ISP1["filetype"],
                start_date=start_date,
            )
        )

    requirements.to_csv("datasets/requirements.csv")
    db.save_df_to_db(dataframe=requirements.copy(), df_name="requirements")

    try:
        start_date = db.get_data('MAX("index")', "units").values[0, 0]
        units = get_unit_data(
            get_excel_data(
                folder_path=configs.UNITS["folder_path"],
                filetype=configs.UNITS["filetype"],
                start_date=start_date,
            )
        )
        if not units.empty:
            units = pd.concat(
                [db.get_data("*", "units")[units.columns], units]
            ).fillna(0)
        else:
            units = db.get_data("*", "units")
    except Exception:
        start_date = "2020-11-01"
        units = get_unit_data(
            get_excel_data(
                folder_path=configs.UNITS["folder_path"],
                filetype=configs.UNITS["filetype"],
                start_date=start_date,
            )
        )

    units.to_csv("datasets/units.csv")
    db.save_df_to_db(dataframe=units.copy(), df_name="units")

    download_weather_data()
    weather = get_weather_mean()
    db.save_df_to_db(dataframe=weather.copy(), df_name="weather")

    try:
        start_date = db.get_data('MAX("index")', "smp").values[0, 0]
        Smp = get_SMP_data(start_date)
        Smp = pd.concat([db.get_data("*", "smp"), Smp])
    except Exception as e:
        print(e)
        Smp = get_SMP_data()

    Smp.to_csv("datasets/SMP.csv")
    db.save_df_to_db(dataframe=Smp.copy(), df_name="smp")

    db = DB("requirements")
    db.save_df_to_db(dataframe=requirements.join(Smp), df_name="requirements")

    db = DB("requirements_units")
    db.save_df_to_db(
        dataframe=requirements.join(units).join(Smp),
        df_name="requirements_units",
    )

    db = DB("requirements_weather")
    db.save_df_to_db(
        dataframe=requirements.join(weather).join(Smp),
        df_name="requirements_weather",
    )

    db = DB("requirements_units_weather")
    db.save_df_to_db(
        dataframe=requirements.join(units).join(weather).join(Smp),
        df_name="requirements_units_weather",
    )
