"""
data/scrapers/nfl.py — NFL data via nfl_data_py.
Pulls schedules, play-by-play (EPA as DVOA proxy), depth charts,
injury designations, NextGen stats, rest days.
"""
import traceback
from typing import Optional

import pandas as pd

from config import MAX_RETRIES


def get_schedules(seasons: list) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        return nfl.import_schedules(seasons)
    except Exception as exc:
        print(f"nfl_data_py schedules error: {exc}")
        return pd.DataFrame()


def get_pbp(seasons: list) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        cols = ["game_id", "posteam", "defteam", "epa", "season", "week",
                "home_team", "away_team", "game_date"]
        df = nfl.import_pbp_data(seasons, columns=cols, downcast=True)
        return df
    except Exception as exc:
        print(f"nfl_data_py pbp error: {exc}")
        return pd.DataFrame()


def get_injuries(seasons: list) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        return nfl.import_injuries(seasons)
    except Exception as exc:
        return pd.DataFrame()


def get_depth_charts(seasons: list) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        return nfl.import_depth_charts(seasons)
    except Exception as exc:
        return pd.DataFrame()


def schedules_to_game_logs(df: pd.DataFrame) -> list:
    """Convert nfl_data_py schedules DataFrame to standardised game log dicts."""
    if df.empty:
        return []
    logs = []
    for _, row in df.iterrows():
        if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
            continue
        home_score = float(row["home_score"])
        away_score = float(row["away_score"])
        logs.append({
            "game_id": str(row.get("game_id", "")),
            "game_date": str(pd.to_datetime(row.get("gameday", "")).date()) if pd.notna(row.get("gameday")) else "",
            "season": int(row.get("season", 0)),
            "week": int(row.get("week", 0)),
            "home_team": str(row.get("home_team", "")),
            "away_team": str(row.get("away_team", "")),
            "home_score": int(home_score),
            "away_score": int(away_score),
            "home_win": home_score > away_score,
            "home_rest_days": int(row["home_rest"]) if pd.notna(row.get("home_rest")) else 7,
            "away_rest_days": int(row["away_rest"]) if pd.notna(row.get("away_rest")) else 7,
            "weather_temp": float(row["temp"]) if pd.notna(row.get("temp")) else None,
            "weather_wind": float(row["wind"]) if pd.notna(row.get("wind")) else None,
            "surface": str(row.get("surface", "")),
            "roof": str(row.get("roof", "")),
            "source": "nfl_data_py",
        })
    return logs


def compute_epa_metrics(pbp_df: pd.DataFrame, team: str, n_games: int = 8) -> dict:
    """EPA-per-play as DVOA proxy for a team over recent games."""
    if pbp_df.empty or "epa" not in pbp_df.columns:
        return {"off_epa": 0.0, "def_epa": 0.0}
    max_plays = n_games * 70
    off = pbp_df[pbp_df["posteam"] == team]["epa"].tail(max_plays)
    def_ = pbp_df[pbp_df["defteam"] == team]["epa"].tail(max_plays)
    return {
        "off_epa": float(off.mean()) if not off.empty else 0.0,
        "def_epa": float(def_.mean()) if not def_.empty else 0.0,
    }


def get_injury_flags(injuries_df: pd.DataFrame, team: str, game_date: str) -> int:
    """Return 1 if team has any Questionable/Doubtful/Out players near game_date."""
    if injuries_df.empty:
        return 0
    bad_statuses = {"Questionable", "Doubtful", "Out", "IR"}
    team_inj = injuries_df[
        (injuries_df["team"] == team) &
        (injuries_df.get("report_status", pd.Series(dtype=str)).isin(bad_statuses))
    ]
    return int(len(team_inj) > 0)


def kaggle_to_game_logs(df: pd.DataFrame) -> list:
    """Convert NFL Kaggle PBP data to game-level logs."""
    if df.empty:
        return []
    try:
        game_cols = ["game_id", "home_team", "away_team", "game_date", "season", "week"]
        available = [c for c in game_cols if c in df.columns]
        games = df[available].drop_duplicates("game_id")
        logs = []
        for _, row in games.iterrows():
            logs.append({
                "game_id": str(row.get("game_id", "")),
                "game_date": str(row.get("game_date", "")),
                "season": int(row.get("season", 0)) if pd.notna(row.get("season")) else 0,
                "week": int(row.get("week", 0)) if pd.notna(row.get("week")) else 0,
                "home_team": str(row.get("home_team", "")),
                "away_team": str(row.get("away_team", "")),
                "home_score": 0,
                "away_score": 0,
                "home_win": False,
                "source": "kaggle",
            })
        return logs
    except Exception:
        return []
