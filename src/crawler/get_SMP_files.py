import os
import re
from datetime import datetime
from os import listdir
from pathlib import Path

import requests


def get_smp_files(last_smp_date: datetime, folder_path: Path):
    if os.path.exists(folder_path):
        print("Directory already exists")
    else:
        print("Creating Directory")
        os.mkdir(folder_path)
    # Search Archive for files if year != current year
    if datetime.now().year != last_smp_date.year:
        pass
    flag = False
    index = 1
    while True:
        url = (
            f"https://www.enexgroup.gr/el/web/guest/markets-publications-el-day-ahead-market?p_p_id"
            f"=com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_9CZslwWTpeD2&p_p_lifecycle=0"
            f"&p_p_state=normal&p_p_mode=view"
            f"&_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_9CZslwWTpeD2_delta=7"
            f"&p_r_p_resetCur=false"
            f"&_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_9CZslwWTpeD2_cur={index}"
        )

        data = requests.get(url)
        print(data)
        x = re.findall("Αποτελέσματα Aγοράς Επόμενης Ημέρας(.* ?)", data.text)
        names = re.findall(
            "[0-9]{8}_EL-DAM_ResultsSummary_EN_v01\\.xlsx", "".join(x)
        )
        urls = re.findall('uuid(.*?)"', "".join(x))
        for i, j in zip(names, urls[: len(names)]):
            file_path = folder_path + i
            if os.path.exists(file_path):
                print(f"File {file_path} found")
                flag = True
            else:
                url = f"https://www.enexgroup.gr/el/c/document_library/get_file?uuid{j}"
                x = requests.get(url).content
                print("Downloading File ")
                with open(file_path, "wb") as xlsx:
                    xlsx.write(x)
        if not names or flag:
            break
        index += 1
    files = [f for f in listdir(folder_path)]
    for name in files:
        if name.endswith("Copy.xlsx"):
            os.remove(folder_path + name)


# https://www.enexgroup.gr/el/c/document_library/get_file?uuid=08c1813a-100c-42cb-0124-d1f6a7eb3325&groupId=20126
