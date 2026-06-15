"""
data/scrapers/mlb.py — MLB data via pybaseball Statcast.
Pulls exit velocity, hard-hit rate, pitcher/batter matchup history,
WAR, park factors, umpire run-environment tendencies.
"""
import traceback
from typing import Optional

import pandas as pd

from config import MAX_RETRIES

PARK_FACTORS = {
    "COL": 1.16, "BOS": 1.09, "CIN": 1.08, "TEX": 1.07, "MIL": 1.05,
    "PHI": 1.04, "HOU": 1.03, "NYY": 1.02, "ATL": 1.01, "CHC": 1.01,
    "LAD": 1.00, "NYM": 1.00, "STL": 0.99, "DET": 0.99, "ARI": 0.99,
    "MIN": 0.98, "CLE": 0.98, "SEA": 0.97, "SFG": 0.97, "TBR": 0.96,
    "BAL": 0.96, "MIA": 0.95, "OAK": 0.95, "SDP": 0.95, "LAA": 0.94,
    "PIT": 0.94, "TOR": 0.94, "KCR": 0.93, "WSN": 0.93, "CHW": 0.92,
}

HIGH_RUN_UMPS = {"bill miller", "dan bellino", "mike everitt", "joe west", "brian gorman"}
LOW_RUN_UMPS = {"angel hernandez", "rob drake", "mark carlson", "alan porter", "paul nauert"}


def get_statcast_range(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Pitch-by-pitch Statcast data for a date range."""
    try:
        from pybaseball import statcast
        df = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception as exc:
        print(f"Statcast error {start_dt}→{end_dt}: {exc}")
        return pd.DataFrame()


def get_statcast_season_chunked(year: int) -> pd.DataFrame:
    """Fetch full season Statcast data in monthly chunks to avoid timeouts."""
    import time
    months = [
        (f"{year}-03-25", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-10-05"),
    ]
    frames = []
    for start, end in months:
        df = get_statcast_range(start, end)
        if not df.empty:
            frames.append(df)
        time.sleep(1.0)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_team_metrics(statcast_df: pd.DataFrame, team: str) -> dict:
    """Exit velocity and hard-hit rate for a team."""
    if statcast_df.empty:
        return {"exit_velocity": 88.0, "hard_hit_rate": 0.35}
    team_df = statcast_df[
        (statcast_df.get("home_team", pd.Series()) == team) |
        (statcast_df.get("away_team", pd.Series()) == team)
    ]
    batted = team_df[team_df["launch_speed"].notna()] if "launch_speed" in team_df.columns else pd.DataFrame()
    if batted.empty:
        return {"exit_velocity": 88.0, "hard_hit_rate": 0.35}
    return {
        "exit_velocity": float(batted["launch_speed"].mean()),
        "hard_hit_rate": float((batted["launch_speed"] >= 95).mean()),
    }


def compute_pitcher_whiff_rate(statcast_df: pd.DataFrame, pitcher_name: str) -> float:
    """Whiff rate (swinging strikes / swings) for a named pitcher."""
    if statcast_df.empty or not pitcher_name or "player_name" not in statcast_df.columns:
        return 0.25
    p_df = statcast_df[
        statcast_df["player_name"].str.lower().str.contains(pitcher_name.lower().split()[-1], na=False)
    ]
    if p_df.empty:
        return 0.25
    swings = p_df[p_df["description"].isin(["swinging_strike", "foul", "hit_into_play",
                                              "swinging_strike_blocked"])]
    whiffs = p_df[p_df["description"].isin(["swinging_strike", "swinging_strike_blocked"])]
    return float(len(whiffs) / len(swings)) if len(swings) > 0 else 0.25


def compute_matchup_win_rate(statcast_df: pd.DataFrame, home_team: str) -> float:
    """Historical home team win rate from Statcast game records."""
    if statcast_df.empty or "home_team" not in statcast_df.columns:
        return 0.5
    home_games = statcast_df[statcast_df["home_team"] == home_team]
    if home_games.empty:
        return 0.5
    by_game = home_games.groupby("game_pk").agg(
        home_score=("home_score", "max"), away_score=("away_score", "max")
    )
    home_wins = (by_game["home_score"] > by_game["away_score"]).mean()
    return float(home_wins) if not pd.isna(home_wins) else 0.5


def get_park_factor(team: str) -> float:
    return PARK_FACTORS.get(team.upper(), 1.0)


def get_umpire_run_factor(umpire: str) -> float:
    name = (umpire or "").lower()
    if any(u in name for u in HIGH_RUN_UMPS):
        return 1.08
    if any(u in name for u in LOW_RUN_UMPS):
        return 0.93
    return 1.0


def statcast_to_game_logs(statcast_df: pd.DataFrame) -> list:
    """Aggregate pitch-by-pitch Statcast into game-level logs for storage."""
    if statcast_df.empty:
        return []
    required = {"game_pk", "game_date", "home_team", "away_team"}
    if not required.issubset(statcast_df.columns):
        return []
    games = statcast_df.groupby("game_pk")
    logs = []
    for game_pk, gdf in games:
        try:
            row = gdf.iloc[0]
            home_team = str(row.get("home_team", ""))
            away_team = str(row.get("away_team", ""))
            home_metrics = compute_team_metrics(gdf, home_team)
            away_metrics = compute_team_metrics(gdf, away_team)
            home_score = int(gdf["home_score"].max()) if "home_score" in gdf.columns else 0
            away_score = int(gdf["away_score"].max()) if "away_score" in gdf.columns else 0
            season = int(row.get("game_year", 0)) if pd.notna(row.get("game_year")) else 0
            logs.append({
                "game_id": f"statcast_{game_pk}",
                "game_date": str(pd.to_datetime(row.get("game_date", "")).date()),
                "season": season,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_score > away_score,
                "home_exit_velocity": home_metrics["exit_velocity"],
                "away_exit_velocity": away_metrics["exit_velocity"],
                "home_hard_hit_rate": home_metrics["hard_hit_rate"],
                "away_hard_hit_rate": away_metrics["hard_hit_rate"],
                "park_factor": get_park_factor(home_team),
                "source": "statcast",
            })
        except Exception:
            continue
    return logs
