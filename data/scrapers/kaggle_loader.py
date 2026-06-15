"""
data/scrapers/kaggle_loader.py — Downloads Kaggle datasets into DataFrames.
Requires KAGGLE_USERNAME and KAGGLE_KEY in Modal Secret kaggle-secret.

Datasets:
  nathanlauga/nba-games            — NBA game logs 2003-2023
  toddsteussie/nfl-play-statistics-dataset-2004-to-present — NFL play-by-play
  wduckett/statcast-data-for-all-mlb-games-2022-23 — MLB Statcast
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


def _auth():
    return (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])


def _download(dataset: str) -> dict:
    """Download a Kaggle dataset zip and return {filename: DataFrame}."""
    url = f"{KAGGLE_BASE}/datasets/download/{dataset}"
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, auth=_auth(), stream=True, timeout=180)
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
    """nathanlauga/nba-games — returns games DataFrame (game-level, not play-by-play)."""
    data = _download(NBA_DATASET)
    # Dataset contains games.csv and games_details.csv
    key = next((k for k in data if "games" in k.lower() and "detail" not in k.lower()), None)
    if not key:
        key = next(iter(data), None)
    df = data[key] if key else pd.DataFrame()
    print(f"[Kaggle NBA] {len(df)} rows from {key}")
    return df


def get_nfl_pbp() -> pd.DataFrame:
    """NFL play-by-play dataset."""
    data = _download(NFL_DATASET)
    key = next(iter(data), None)
    df = data[key] if key else pd.DataFrame()
    print(f"[Kaggle NFL] {len(df)} rows from {key}")
    return df


def get_mlb_statcast() -> pd.DataFrame:
    """MLB Statcast pitch-by-pitch dataset."""
    data = _download(MLB_DATASET)
    key = next(iter(data), None)
    df = data[key] if key else pd.DataFrame()
    print(f"[Kaggle MLB] {len(df)} rows from {key}")
    return df
