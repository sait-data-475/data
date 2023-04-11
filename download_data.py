import tarfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi

output = Path("output")
output.mkdir(exist_ok=True)

api = KaggleApi()
api.authenticate()

api.competition_download_files(
    "titanic",
    path=output,
)
api.competition_download_files(
    "house-prices-advanced-regression-techniques",
    path=output,
)
api.competition_download_files(
    "plant-seedlings-classification",
    path=output,
)

base = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

r = requests.get(urljoin(base, "00236/seeds_dataset.txt"))
with open(output / "seeds_dataset.txt", "wb") as f:
    f.write(r.content)

r = requests.get(urljoin(base, "00352/Online%20Retail.xlsx"))
with open(output / "Online Retail.xlsx", "wb") as f:
    f.write(r.content)

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
r = requests.get(url)
with open(output / "aclImdb_v1.tar.gz", "wb") as f:
    f.write(r.content)


with tarfile.open(output / "aclImdb_v1.tar.gz") as f:
    f.extractall(output)

imdb = output / "imdb"
imdb.mkdir(exist_ok=True)


def get_text(path):
    with open(path, encoding="utf-8") as f:
        text = BeautifulSoup(f, features="html.parser").get_text()

    return text


def create_dataset(name):
    review, sentiment = [], []
    pos_docs = Path(output / f"aclImdb/{name}/pos")
    for path in tqdm(list(pos_docs.glob("*.txt")), desc=f"{name} pos"):
        text = get_text(path)
        sentiment.append("positive")
        review.append(text)

    neg_docs = Path(output / f"aclImdb/{name}/neg")
    for path in tqdm(list(neg_docs.glob("*.txt")), desc=f"{name} neg"):
        text = get_text(path)
        sentiment.append("negative")
        review.append(text)

    pd.DataFrame(dict(review=review, sentiment=sentiment)).to_csv(
        imdb / f"{name}.csv", index=False
    )


create_dataset("train")
create_dataset("test")
