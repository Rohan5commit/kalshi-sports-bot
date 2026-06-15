"""
data/scrapers/nba.py — NBA data via nba_api.
Pulls box scores, advanced metrics (TS%, USG%), rest days,
back-to-back flags, referee assignments, and home/away splits.
"""
import time
import traceback
from datetime import date
from typing import Optional

import pandas as pd

from config import MAX_RETRIES, BACKOFF_BASE

# nba_api rate-limit: 1 req / ~0.6s
_NBA_DELAY = 0.65


def _call(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            result = fn(*args, **kwargs)
            time.sleep(_NBA_DELAY)
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
    raise last_exc


def get_season_games(season: int) -> list:
    """
    All NBA regular-season games for a given season year (e.g. 2023 → 2023-24).
    Returns list of game dicts with home/away teams, scores, date.
    """
    from nba_api.stats.endpoints import LeagueGameFinder
    season_str = f"{season}-{str(season + 1)[-2:]}"
    finder = _call(LeagueGameFinder,
                   season_nullable=season_str,
                   league_id_nullable="00",
                   season_type_nullable="Regular Season")
    df = finder.get_data_frames()[0]

    games: dict = {}
    for _, row in df.iterrows():
        gid = str(row["GAME_ID"])
        matchup = str(row.get("MATCHUP", ""))
        is_home = "@" not in matchup

        if gid not in games:
            games[gid] = {"game_id": gid, "game_date": str(row["GAME_DATE"]), "season": season}

        side = "home" if is_home else "away"
        games[gid][f"{side}_team"] = row["TEAM_ABBREVIATION"]
        games[gid][f"{side}_score"] = int(row.get("PTS") or 0)
        games[gid][f"{side}_win"] = row.get("WL") == "W"

    result = []
    for g in games.values():
        if "home_team" in g and "away_team" in g:
            g["home_win"] = bool(g.get("home_win", False))
            result.append(g)
    return result


def add_rest_and_b2b(games: list) -> list:
    """Compute rest_days and back_to_back for each team in each game (in-place)."""
    last: dict[str, date] = {}
    for g in sorted(games, key=lambda x: x["game_date"]):
        gdate = pd.to_datetime(g["game_date"]).date()
        for side in ("home", "away"):
            team = g.get(f"{side}_team")
            if not team:
                continue
            if team in last:
                rest = (gdate - last[team]).days
                g[f"{side}_rest_days"] = rest
                g[f"{side}_back_to_back"] = rest <= 1
            else:
                g[f"{side}_rest_days"] = 7
                g[f"{side}_back_to_back"] = False
            last[team] = gdate
    return games


def get_advanced_stats(game_id: str) -> dict:
    """TS% and USG% (starter averages) for home and away teams."""
    try:
        from nba_api.stats.endpoints import BoxScoreAdvancedV2
        adv = _call(BoxScoreAdvancedV2, game_id=game_id)
        df = adv.get_data_frames()[0]
        starters = df[df["START_POSITION"].notna() & (df["START_POSITION"] != "")]
        if starters.empty:
            return {}
        # Split by team (first 5 = home starters in nba_api order by default — use TEAM_ID grouping)
        teams = starters["TEAM_ID"].unique()
        if len(teams) < 2:
            return {}
        h_df = starters[starters["TEAM_ID"] == teams[0]]
        a_df = starters[starters["TEAM_ID"] == teams[1]]
        return {
            "home_ts_pct": float(h_df["TS_PCT"].mean()),
            "away_ts_pct": float(a_df["TS_PCT"].mean()),
            "home_usg_pct": float(h_df["USG_PCT"].mean()),
            "away_usg_pct": float(a_df["USG_PCT"].mean()),
        }
    except Exception:
        return {}


def get_officials(game_id: str) -> str:
    """Return comma-joined official names for a game."""
    try:
        from nba_api.stats.endpoints import BoxScoreSummaryV2
        summary = _call(BoxScoreSummaryV2, game_id=game_id)
        officials_df = summary.get_data_frames()[2]
        return ", ".join(officials_df["OFFICIAL_NAME"].tolist())
    except Exception:
        return ""


def get_home_away_win_rates(team_abbr: str, season: int) -> dict:
    """Win rate at home vs away for a team in a season."""
    try:
        from nba_api.stats.endpoints import LeagueGameFinder
        season_str = f"{season}-{str(season + 1)[-2:]}"
        finder = _call(LeagueGameFinder, team_abbreviation_nullable=team_abbr,
                       season_nullable=season_str)
        df = finder.get_data_frames()[0]
        home_df = df[~df["MATCHUP"].str.contains("@")]
        away_df = df[df["MATCHUP"].str.contains("@")]
        return {
            "home_win_rate": float((home_df["WL"] == "W").mean()) if len(home_df) else 0.5,
            "away_win_rate": float((away_df["WL"] == "W").mean()) if len(away_df) else 0.5,
        }
    except Exception:
        return {"home_win_rate": 0.5, "away_win_rate": 0.5}


def enrich_games(games: list, fetch_advanced: bool = False) -> list:
    """
    Add advanced stats and officials to game dicts.
    fetch_advanced=True hits nba_api per-game (slow, rate-limited).
    """
    for g in games:
        if fetch_advanced:
            try:
                adv = get_advanced_stats(g["game_id"])
                g.update(adv)
                g["referee"] = get_officials(g["game_id"])
            except Exception:
                pass
    return games


def kaggle_to_game_logs(df: pd.DataFrame) -> list:
    """Convert nathanlauga/nba-games DataFrame to standardised game log dicts."""
    if df.empty:
        return []
    logs = []
    for _, row in df.iterrows():
        try:
            home_win = bool(row.get("HOME_TEAM_WINS", 0))
            logs.append({
                "game_id": f"kaggle_nba_{row.get('GAME_ID', '')}",
                "game_date": str(pd.to_datetime(row["GAME_DATE_EST"]).date()),
                "season": int(row.get("SEASON", 0)),
                "home_team": str(row.get("HOME_TEAM_ID", "")),
                "away_team": str(row.get("VISITOR_TEAM_ID", "")),
                "home_score": int(row.get("PTS_home", 0) or 0),
                "away_score": int(row.get("PTS_away", 0) or 0),
                "home_win": home_win,
                "source": "kaggle",
            })
        except Exception:
            continue
    return logs
