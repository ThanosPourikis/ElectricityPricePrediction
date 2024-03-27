import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import feedparser
import requests
from bs4 import BeautifulSoup
from sqlmodel import Session, create_engine, SQLModel

from configs import config
from data.get_SMP_data import parse_xlsx_file
from orm.smp import SMPModel, Dam, Lida1, Lida2, Lida3


def get_archived_files(folder_path: Path):
    url = "https://www.enexgroup.gr/el/dam-idm-archive"
    down_url = "https://www.enexgroup.gr/"
    data = requests.get(url)
    soup = BeautifulSoup(data.content, "html.parser")
    for item in soup.find_all("i", class_="icon-download"):
        file_path = folder_path / str(item.next_sibling).strip()
        if config.DAM_PATTERN in item.next_sibling:
            if not file_path.exists():
                with open(file_path, "wb") as zip_file:
                    zip_file.write(
                        requests.get(
                            down_url + item.parent.attrs["href"]
                        ).content
                    )
            print("Processing Zip file")
            with ZipFile(file_path, "r") as zip_file:
                zip_file.extractall(folder_path / "unzipped")


def upload_data(folders, unzip_path, engine, model: SMPModel):
    for j in folders:
        df = parse_xlsx_file(
            [unzip_path / j / i for i in os.listdir(unzip_path / j)]
        )
        with Session(engine) as session:
            for i in df.to_dict("records"):
                session.add(model(timestamp=i["Date"], value=i["SMP"]))
            session.commit()


def main():


    url = "https://www.enexgroup.gr/el/web/guest/markets-publications-el-day-ahead-market?p_p_id=com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_9CZslwWTpeD2&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_resource_id=getRSS&p_p_cacheability=cacheLevelPage"
    feed = feedparser.parse(url)

    print(feed)

    get_archived_files(config.FOLDER_PATH)
    unzip_path = config.FOLDER_PATH / "unzipped"
    dirs = set(os.listdir(unzip_path))
    engine = create_engine(
        "postgresql://fl0user:jpwK5XHd3oWL@ep-damp-poetry-a24vlps7.eu-central-1.aws.neon.fl0.io:5432/ElectricityPricePrediction"
    )

    SQLModel.metadata.create_all(engine)

    dam_folders = [i for i in dirs if "dam" in i.lower()]
    lida1 = [i for i in dirs if "a1" in i.lower()]
    lida2 = [i for i in dirs if "a2" in i.lower()]
    lida3 = [i for i in dirs if "a3" in i.lower()]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for folder, model in zip(
            [dam_folders, lida1, lida2, lida3], [Dam, Lida1, Lida2, Lida3]
        ):
            executor.submit(upload_data, folder, unzip_path, engine, model)


if __name__ == "__main__":
    main()
