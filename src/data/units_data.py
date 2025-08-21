from datetime import datetime, timedelta

import pandas as pd
from pytz import timezone

localTz = timezone("CET")


def get_unit_data(df):
    export = pd.DataFrame()

    for i in df:
        temp = pd.DataFrame(columns=df[i].iloc[3:, 1].values)
        date = datetime.fromisoformat(f"{i[0:4]}-{i[4:6]}-{i[6:8]}")
        date = [date + timedelta(hours=i + 1) for i in range(24)]
        date = [localTz.localize(x) for x in date]
        for j in range(3, len(df[i])):
            temp.loc[0, df[i].iloc[j].iloc[1]] = df[i].iloc[j].iloc[3]
        x = temp
        for _ in range(23):
            temp = temp.append(x)
        temp.index = date

        export = pd.concat([export, temp])
    # export = export.dropna(axis=1)
    export = export.fillna(0)
    export = export.sort_index()
    export["Date"] = export.index
    export = export.set_index("Date")
    return export
