"""
data/scrapers/kaggle_loader.py — Downloads Kaggle datasets into DataFrames.
Uses KAGGLE_API_TOKEN (new bearer token format, KGAT_...) from kaggle-secret.

Datasets:
  nathanlauga/nba-games
  toddsteussie/nfl-play-statistics-dataset-2004-to-present
  wduckett/statcast-data-for-all-mlb-games-2022-23
"""
import io
import os
import time
import zipfile

import pandas as pd
import requests

from config import MAX_RETRIES, BACKOFF_BASE

KAGGLE_BASE = "https://www.kaggle.com/api/v1"

NBA_DATASET = "nathanlauga/nba-games"
NFL_DATASET = "toddsteussie/nfl-play-statistics-dataset-2004-to-present"
MLB_DATASET = "wduckett/statcast-data-for-all-mlb-games-2022-23"


def _headers() -> dict:
    token = os.environ["KAGGLE_API_TOKEN"]
    return {"Authorization": f"Bearer {token}"}


def _download(dataset: str) -> dict:
    """Download a Kaggle dataset zip and return {filename: DataFrame}."""
    url = f"{KAGGLE_BASE}/datasets/download/{dataset}"
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=_headers(), stream=True, timeout=300)
            resp.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            return {
                name: pd.read_csv(z.open(name), low_memory=False)
                for name in z.namelist()
                if name.endswith(".csv")
            }
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
    raise last_exc


def get_nba_games() -> pd.DataFrame:
    data = _download(NBA_DATASET)
    key = next((k for k in data if "games" in k.lower() and "detail" not in k.lower()), None)
    if not key:
        key = next(iter(data), None)
    df = data[key] if key else pd.DataFrame()
    print(f"[Kaggle NBA] {len(df)} rows from {key}")
    return df


def get_nfl_pbp() -> pd.DataFrame:
    data = _download(NFL_DATASET)
    key = next(iter(data), None)
    df = data[key] if key else pd.DataFrame()
    print(f"[Kaggle NFL] {len(df)} rows from {key}")
    return df


def get_mlb_statcast() -> pd.DataFrame:
    data = _download(MLB_DATASET)
    key = next(iter(data), None)
    df = data[key] if key else pd.DataFrame()
    print(f"[Kaggle MLB] {len(df)} rows from {key}")
    return df
