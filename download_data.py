from pathlib import Path
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi

output = Path("output")
output.mkdir(exist_ok=True)

api = KaggleApi()
api.authenticate()

api.competition_download_files(
    "titanic",
    path=output,
    quiet=True,
)
api.competition_download_files(
    "house-prices-advanced-regression-techniques",
    path=output,
    quiet=True,
)

base = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

r = requests.get(urljoin(base, "00236/seeds_dataset.txt"))
with open(output / "seeds_dataset.txt", "wb") as f:
    f.write(r.content)

r = requests.get(urljoin(base, "00352/Online%20Retail.xlsx"))
with open(output / "Online Retail.xlsx", "wb") as f:
    f.write(r.content)

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
with open(output / "aclImdb_v1.tar.gz", "wb") as f:
    f.write(r.content)
