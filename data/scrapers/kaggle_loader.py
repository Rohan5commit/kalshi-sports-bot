"""
data/scrapers/kaggle_loader.py — Downloads Kaggle datasets into DataFrames.
Uses KAGGLE_API_TOKEN (bearer token KGAT_...) from kaggle-secret.

Datasets:
  NBA: nathanlauga/nba-games, wyattowalsh/basketball,
       sumitrodatta/nba-aba-baa-stats, dansbecker/nba-shot-logs
  NFL: tobycrabtree/nfl-scores-and-betting-data,
       maxhorowitz/nflplaybyplay2009to2016
  MLB: vivovinco/19622022-mlb-season-stats,
       saurabhshahane/mlb-game-log-dataset
"""
import io
import os
import time
import zipfile

import pandas as pd
import requests

from config import MAX_RETRIES, BACKOFF_BASE

KAGGLE_BASE = "https://www.kaggle.com/api/v1"

# Dataset identifiers
_NBA_GAMES_DS = "nathanlauga/nba-games"
_NBA_FULL_DS = "wyattowalsh/basketball"
_NBA_STATS_DS = "sumitrodatta/nba-aba-baa-stats"
_NBA_SHOTS_DS = "dansbecker/nba-shot-logs"
_NFL_SCORES_DS = "tobycrabtree/nfl-scores-and-betting-data"
_NFL_PBP_DS = "maxhorowitz/nflplaybyplay2009to2016"
_MLB_SEASON_DS = "vivovinco/19622022-mlb-season-stats"
_MLB_GAMELOG_DS = "saurabhshahane/mlb-game-log-dataset"


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
            result = {}
            for name in z.namelist():
                if name.endswith(".csv"):
                    try:
                        result[name] = pd.read_csv(z.open(name), low_memory=False)
                    except Exception as e:
                        print(f"  [Kaggle] Could not parse {name}: {e}")
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
    raise last_exc


def _first_df(data: dict, hint: str = "") -> pd.DataFrame:
    """Pick the best-matching CSV from a Kaggle zip."""
    if not data:
        return pd.DataFrame()
    if hint:
        key = next((k for k in data if hint.lower() in k.lower()), None)
        if key:
            return data[key]
    return data[next(iter(data))]


# ── NBA datasets ───────────────────────────────────────────────────────────────

def get_nba_games() -> pd.DataFrame:
    """nathanlauga/nba-games — game-level box scores, 2004-2022."""
    data = _download(_NBA_GAMES_DS)
    key = next((k for k in data if "games" in k.lower() and "detail" not in k.lower()), None)
    df = data[key] if key else _first_df(data)
    print(f"[Kaggle NBA games] {len(df)} rows")
    return df


def get_nba_full() -> dict:
    """
    wyattowalsh/basketball — comprehensive NBA dataset including:
    game.csv, line_score.csv, player_info.csv, team_info_common.csv, etc.
    Returns the full dict of DataFrames.
    """
    data = _download(_NBA_FULL_DS)
    print(f"[Kaggle wyattowalsh] files: {list(data.keys())}")
    return data


def get_nba_stats_history() -> pd.DataFrame:
    """sumitrodatta/nba-aba-baa-stats — season stats back to 1947."""
    data = _download(_NBA_STATS_DS)
    df = _first_df(data, "player")
    if df.empty:
        df = _first_df(data)
    print(f"[Kaggle NBA stats history] {len(df)} rows")
    return df


def get_nba_shot_logs() -> pd.DataFrame:
    """dansbecker/nba-shot-logs — shot log data 2014-15 season."""
    data = _download(_NBA_SHOTS_DS)
    df = _first_df(data, "shot")
    print(f"[Kaggle NBA shot logs] {len(df)} rows")
    return df


# ── NFL datasets ───────────────────────────────────────────────────────────────

def get_nfl_scores() -> pd.DataFrame:
    """tobycrabtree/nfl-scores-and-betting-data — historical NFL scores 1966-2020."""
    data = _download(_NFL_SCORES_DS)
    df = _first_df(data, "score")
    print(f"[Kaggle NFL scores] {len(df)} rows")
    return df


def get_nfl_pbp() -> pd.DataFrame:
    """maxhorowitz/nflplaybyplay2009to2016 — play-by-play 2009-2016."""
    data = _download(_NFL_PBP_DS)
    df = _first_df(data, "play")
    print(f"[Kaggle NFL PBP] {len(df)} rows")
    return df


# ── MLB datasets ───────────────────────────────────────────────────────────────

def get_mlb_season_stats() -> pd.DataFrame:
    """vivovinco/19622022-mlb-season-stats — MLB team/player season stats 1962-2022."""
    data = _download(_MLB_SEASON_DS)
    df = _first_df(data)
    print(f"[Kaggle MLB season stats] {len(df)} rows")
    return df


def get_mlb_game_logs() -> pd.DataFrame:
    """saurabhshahane/mlb-game-log-dataset — game-level logs 1871-present."""
    data = _download(_MLB_GAMELOG_DS)
    df = _first_df(data, "game")
    print(f"[Kaggle MLB game logs] {len(df)} rows")
    return df


# ── Conversion utilities ───────────────────────────────────────────────────────

def nfl_scores_to_game_logs(df: pd.DataFrame) -> list:
    """Convert tobycrabtree NFL scores DataFrame to standard game log dicts."""
    if df.empty:
        return []
    logs = []
    # Possible column names
    h_col = next((c for c in df.columns if "home" in c.lower() and "team" in c.lower()), None)
    a_col = next((c for c in df.columns if ("away" in c.lower() or "visit" in c.lower()) and "team" in c.lower()), None)
    hs_col = next((c for c in df.columns if "home" in c.lower() and ("score" in c.lower() or "pts" in c.lower())), None)
    as_col = next((c for c in df.columns if ("away" in c.lower() or "visit" in c.lower()) and ("score" in c.lower() or "pts" in c.lower())), None)
    date_col = next((c for c in df.columns if "date" in c.lower() or "schedule_date" in c.lower()), None)

    if not all([h_col, a_col, hs_col, as_col, date_col]):
        return []

    for _, row in df.iterrows():
        try:
            h_pts = int(float(row.get(hs_col) or 0))
            a_pts = int(float(row.get(as_col) or 0))
            game_date = str(row[date_col])[:10]
            season = int(game_date[:4]) if game_date else 0
            logs.append({
                "game_id": f"kaggle_nfl_{hash(str(row[date_col]) + str(row[h_col]))}",
                "game_date": game_date,
                "season": season,
                "home_team": str(row[h_col]),
                "away_team": str(row[a_col]),
                "home_score": h_pts,
                "away_score": a_pts,
                "home_win": h_pts > a_pts,
                "source": "kaggle_nfl_scores",
            })
        except Exception:
            continue
    return [g for g in logs if g["game_date"] and g["home_team"] and g["away_team"]]


def mlb_game_logs_to_standard(df: pd.DataFrame) -> list:
    """Convert saurabhshahane MLB game log DataFrame to standard dicts."""
    if df.empty:
        return []
    # Possible column patterns in this dataset
    h_col = next((c for c in df.columns if "h_name" in c.lower() or ("home" in c.lower() and "team" in c.lower())), None)
    v_col = next((c for c in df.columns if "v_name" in c.lower() or ("visit" in c.lower() and "team" in c.lower()) or "away" in c.lower()), None)
    hs_col = next((c for c in df.columns if "h_score" in c.lower() or ("home" in c.lower() and "score" in c.lower())), None)
    vs_col = next((c for c in df.columns if "v_score" in c.lower() or ("visit" in c.lower() and "score" in c.lower()) or ("away" in c.lower() and "score" in c.lower())), None)
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if not all([h_col, v_col, hs_col, vs_col, date_col]):
        return []
    logs = []
    for _, row in df.iterrows():
        try:
            h_pts = int(float(row.get(hs_col) or 0))
            v_pts = int(float(row.get(vs_col) or 0))
            game_date = str(row[date_col])[:10]
            if len(game_date) == 8 and game_date.isdigit():
                game_date = f"{game_date[:4]}-{game_date[4:6]}-{game_date[6:]}"
            season = int(game_date[:4]) if game_date and len(game_date) >= 4 else 0
            logs.append({
                "game_id": f"kaggle_mlb_{hash(game_date + str(row[h_col]))}",
                "game_date": game_date,
                "season": season,
                "home_team": str(row[h_col]),
                "away_team": str(row[v_col]),
                "home_score": h_pts,
                "away_score": v_pts,
                "home_win": h_pts > v_pts,
                "source": "kaggle_mlb_logs",
            })
        except Exception:
            continue
    return [g for g in logs if g["game_date"] and g["home_team"] and g["away_team"]]
