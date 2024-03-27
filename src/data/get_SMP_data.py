from pathlib import Path
from typing import List

from crawler.get_SMP_files import get_smp_files
from datetime import timedelta, datetime
from os import listdir
import pandas as pd
from pytz import timezone
from zoneinfo import ZoneInfo

localTz = timezone("CET")
dt = localTz.localize(datetime.now()) + timedelta(days=1)


def get_SMP_data(start_date, folder_path):
    files = []
    if start_date is None:
        files = [f for f in listdir(folder_path)]
    else:
        start_date = datetime.fromisoformat(start_date)
        days = (dt - start_date).days
        for i in range(days):
            date = start_date + timedelta(days=i)
            files.append(
                f"{str(date)[:10].replace('-', '')}_EL-DAM_ResultsSummary_EN_v01.xlsx"
            )


def parse_xlsx_file(files: List[Path]) -> pd.DataFrame:
    export = pd.DataFrame()
    for file in files:
        df = pd.read_excel(file, header=None)
        line = (df[df.iloc[:, 0] == "Market Clearing Price"]).index[0] + 1
        temp = (
            pd.DataFrame(df.iloc[line][1:-1]).dropna().reset_index(drop=True)
        )

        temp["Date"] = (
            pd.to_datetime(df.iloc[1][0])
            .tz_localize("Europe/Athens")
            .tz_convert("UTC")
        )
        temp["Date"] += pd.to_timedelta(
            df.iloc[1][1:-2].reset_index(drop=True), "h"
        )
        print(f"Proccessing {file}")
        export = pd.concat([export, temp])

    export = export.drop_duplicates().set_index("Date")
    export.columns = ["SMP"]
    export["SMP"] = export["SMP"].astype(float)
    export = export.sort_index()
    export = export.reset_index()
    export["Date"] = export["Date"].dt.tz_localize(None)
    return export
