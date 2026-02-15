from datetime import datetime, timedelta
import pandas as pd
from pytz import timezone

localTz = timezone("CET")


def get_isp_data(df):
    export_df = pd.DataFrame()
    for i in df:
        temp = pd.DataFrame()

        date = datetime.fromisoformat(
            f"{i[0:4]}-{i[4:6]}-{i[6:8]}"
        ) + timedelta(hours=1)
        date = [date + timedelta(hours=i) for i in range(24)]
        date = [localTz.localize(x) for x in date]
        # del date[3]

        res = df[i].iloc[
            df[i][df[i].iloc[:, 0] == "Non-Dispatchable RES"].index[0] + 1
        ][2:-1]
        temp["Renewables"] = res

        nndis = df[i].iloc[
            df[i][df[i].iloc[:, 0] == "Non-Dispatcheble Losses"].index[0] + 1
        ]
        temp["Non-Dispatcheble"] = nndis

        man_hydro = df[i].iloc[
            df[i][df[i].iloc[:, 0] == "Commissioning"].index[0] - 1
        ]
        temp["Man_Hydro"] = man_hydro

        comm = df[i].iloc[
            df[i][df[i].iloc[:, 0] == "Commissioning"].index[0] + 1
        ]
        temp["Commissioning"] = comm

        req_start = df[i][df[i].iloc[:, 0] == "Up"].index[0]
        req_end = req_start + 3

        temp["Reserve_Up"] = df[i].iloc[req_start:req_end, :-1].sum()[2:-1]

        req_start = df[i][df[i].iloc[:, 0] == "Down"].index[0]
        req_end = req_start + 3

        temp["Reserve_Down"] = df[i].iloc[req_start:req_end, :-1].sum()[2:-1]

        pairs = []
        for j in range(0, len(temp), 2):
            pairs.append(
                temp.iloc[j : j + 2].mean(axis=0)
            )  # Example (00:00:00 + 00:30:00)/2 -> 00:00:00
        pairs = pd.DataFrame(pairs)

        # if len(pairs) == 23:
        # 	#Daylight saving 02:00:00 is missing so i aproximate the values from the previous and next hours
        # 	pairs = pairs.append(pd.DataFrame(pairs.iloc[1:3].mean(axis=0).to_dict(),index=[1.5])).sort_index().reset_index(drop = True)
        # 	pairs["Date"] = date
        # 	del pairs
        # elif len(pairs) == 25:
        # 	#Daylight saving 25hours mean
        # 	pairs.iloc[3] = pairs.iloc[1:3].mean(axis=0)
        # 	pairs = pairs.drop([4]).sort_index().reset_index(drop = True)
        # 	pairs["Date"] = date

        if len(pairs) == 24:
            pairs["Date"] = date
            pairs = pairs.set_index("Date")
            export_df = export_df.append(pairs)

    export_df = export_df.sort_index()
    return export_df
