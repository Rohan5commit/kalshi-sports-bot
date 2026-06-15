"""
db/historical_store.py — Supabase storage for historical game logs and Elo ratings.
Pattern:
  First run  → full load from all sources
  Subsequent → check MAX(game_date), fetch only newer data
"""
import os
from datetime import date, datetime
from typing import Optional

from supabase import create_client, Client

_TABLE = {"NBA": "nba_game_logs", "NFL": "nfl_game_logs", "MLB": "mlb_game_logs"}


def _sb() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def get_last_game_date(sport: str) -> Optional[date]:
    """Most recent game_date stored for a sport, or None if table is empty."""
    try:
        res = _sb().table(_TABLE[sport]).select("game_date").order("game_date", desc=True).limit(1).execute()
        if res.data:
            return datetime.strptime(res.data[0]["game_date"], "%Y-%m-%d").date()
    except Exception:
        pass
    return None


def upsert_game_logs(sport: str, logs: list):
    """Upsert list of game log dicts into the sport table. game_id is the PK."""
    if not logs:
        return
    sb = _sb()
    table = _TABLE[sport]
    for i in range(0, len(logs), 500):
        chunk = logs[i:i + 500]
        # Remove None values to avoid Supabase type errors
        clean = [{k: v for k, v in row.items() if v is not None} for row in chunk]
        try:
            sb.table(table).upsert(clean, on_conflict="game_id").execute()
            print(f"[{sport}] upserted {len(clean)} logs (chunk {i // 500})")
        except Exception as exc:
            print(f"upsert_game_logs error chunk {i}: {exc}")


def get_game_logs(sport: str, since_date: Optional[date] = None, limit: int = 10000) -> list:
    """Retrieve stored game logs sorted by date ascending."""
    try:
        q = _sb().table(_TABLE[sport]).select("*").order("game_date").limit(limit)
        if since_date:
            q = q.gte("game_date", str(since_date))
        return q.execute().data
    except Exception as exc:
        print(f"get_game_logs error: {exc}")
        return []


def upsert_elo_ratings(sport: str, ratings: dict):
    """Upsert {team: elo} dict into elo_ratings table."""
    if not ratings:
        return
    today = str(datetime.utcnow().date())
    records = [
        {"sport": sport, "team": team, "elo": round(elo, 2), "last_updated": today}
        for team, elo in ratings.items()
    ]
    try:
        _sb().table("elo_ratings").upsert(records, on_conflict="sport,team").execute()
    except Exception as exc:
        print(f"upsert_elo_ratings error: {exc}")


def get_elo_ratings(sport: str) -> dict:
    """Return {team: elo} for all teams in a sport."""
    try:
        res = _sb().table("elo_ratings").select("team,elo").eq("sport", sport).execute()
        return {r["team"]: r["elo"] for r in res.data}
    except Exception:
        return {}
