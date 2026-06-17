"""
data/scrapers/nba_data.py — Pull NBA game logs via nba_api (stats.nba.com).
Uses LeagueGameLog endpoint — not blocked from cloud IPs (unlike ESPN API).
"""
import time
from datetime import datetime
from typing import Optional

import pandas as pd


def _season_str(season: int) -> str:
    return f"{season}-{str(season + 1)[-2:]}"


def _parse_nba_date(d: str) -> str:
    """Convert 'NOV 01, 2023' or '2023-11-01' to 'YYYY-MM-DD'."""
    if not d:
        return ""
    d = str(d).strip()
    if len(d) == 10 and d[4] == "-":
        return d
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(d, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return d[:10]


def _get_game_log(season: int, season_type: str = "Regular Season") -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueGameLog
    try:
        time.sleep(0.8)
        lg = LeagueGameLog(
            season=_season_str(season),
            season_type_all_star=season_type,
            timeout=60,
        )
        return lg.get_data_frames()[0]
    except Exception as exc:
        print(f"[NBA] nba_api {season} {season_type}: {exc}")
        return pd.DataFrame()


def _df_to_records(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    records: dict = {}
    for _, row in df.iterrows():
        gid = str(row.get("GAME_ID", ""))
        if not gid:
            continue
        matchup = str(row.get("MATCHUP", ""))
        abbr = str(row.get("TEAM_ABBREVIATION", ""))
        is_home = "vs." in matchup
        pts = int(row.get("PTS") or 0)
        wl = str(row.get("WL") or "")
        game_date = _parse_nba_date(str(row.get("GAME_DATE", "")))

        if gid not in records:
            records[gid] = {
                "game_id": f"nba_api_{gid}",
                "game_date": game_date,
                "home_team": "", "away_team": "",
                "home_score": 0, "away_score": 0,
                "home_win": False,
            }
        r = records[gid]
        if is_home:
            r["home_team"] = abbr
            r["home_score"] = pts
            r["home_win"] = (wl == "W")
            r["game_date"] = game_date
        else:
            r["away_team"] = abbr
            r["away_score"] = pts

    return [r for r in records.values() if r["home_team"] and r["away_team"]]


def ingest_full_history(seasons: list, incremental_from=None) -> list:
    all_logs: list = []
    for season in seasons:
        reg = _df_to_records(_get_game_log(season, "Regular Season"))
        po = _df_to_records(_get_game_log(season, "Playoffs"))
        season_logs = reg + po
        if incremental_from:
            season_logs = [g for g in season_logs
                           if g.get("game_date", "") > str(incremental_from)]
        all_logs.extend(season_logs)
        print(f"[NBA] nba_api {season}: {len(reg)} regular + {len(po)} playoff")
        time.sleep(0.3)
    return all_logs
