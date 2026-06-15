"""
db/historical_store.py — Supabase storage for historical game logs, Elo ratings,
lineup stats, referee tendencies, pitcher-batter matchups, weather cache.
Pattern: full load on first run; incremental on subsequent runs.
"""
import os
from datetime import date, datetime
from typing import Optional

from supabase import create_client, Client

_GAME_TABLES = {"NBA": "nba_game_logs", "NFL": "nfl_game_logs", "MLB": "mlb_game_logs"}


def _sb() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def _clean(rows: list) -> list:
    """Strip None values to avoid Supabase type errors."""
    return [{k: v for k, v in row.items() if v is not None} for row in rows]


# ── Game logs ──────────────────────────────────────────────────────────────────

def get_last_game_date(sport: str) -> Optional[date]:
    """Most recent game_date stored for a sport."""
    try:
        res = (_sb().table(_GAME_TABLES[sport])
               .select("game_date").order("game_date", desc=True).limit(1).execute())
        if res.data:
            return datetime.strptime(res.data[0]["game_date"], "%Y-%m-%d").date()
    except Exception:
        pass
    return None


def upsert_game_logs(sport: str, logs: list):
    """Upsert game log dicts. game_id is the conflict key."""
    if not logs:
        return
    sb = _sb()
    table = _GAME_TABLES[sport]
    for i in range(0, len(logs), 500):
        chunk = _clean(logs[i:i + 500])
        try:
            sb.table(table).upsert(chunk, on_conflict="game_id").execute()
            print(f"[{sport}] upserted {len(chunk)} logs (chunk {i // 500})")
        except Exception as exc:
            print(f"upsert_game_logs error chunk {i}: {exc}")


def get_game_logs(sport: str, since_date: Optional[date] = None,
                  limit: int = 10000) -> list:
    """Retrieve stored game logs sorted by date ascending."""
    try:
        q = _sb().table(_GAME_TABLES[sport]).select("*").order("game_date").limit(limit)
        if since_date:
            q = q.gte("game_date", str(since_date))
        return q.execute().data
    except Exception as exc:
        print(f"get_game_logs error: {exc}")
        return []


# ── Elo ratings ────────────────────────────────────────────────────────────────

def upsert_elo_ratings(sport: str, ratings: dict):
    """Upsert {team: elo} into elo_ratings table."""
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
    """Return {team: elo} for a sport."""
    try:
        res = _sb().table("elo_ratings").select("team,elo").eq("sport", sport).execute()
        return {r["team"]: r["elo"] for r in res.data}
    except Exception:
        return {}


# ── NBA lineup stats ───────────────────────────────────────────────────────────

def upsert_lineup_stats(sport: str, records: list):
    """
    Upsert 5-man lineup net ratings.
    Each record: {id, sport, season, team, lineup, net_rating, possessions}.
    """
    if not records:
        return
    sb = _sb()
    for i in range(0, len(records), 500):
        chunk = _clean(records[i:i + 500])
        try:
            sb.table("nba_lineup_stats").upsert(chunk, on_conflict="id").execute()
        except Exception as exc:
            print(f"upsert_lineup_stats error: {exc}")


def get_lineup_net_rating(team: str, season: int) -> float:
    """Best 5-man lineup net rating for a team in a season."""
    try:
        res = (_sb().table("nba_lineup_stats")
               .select("net_rating").eq("team", team).eq("season", season)
               .order("possessions", desc=True).limit(1).execute())
        if res.data:
            return float(res.data[0]["net_rating"])
    except Exception:
        pass
    return 0.0


# ── Referee tendencies ─────────────────────────────────────────────────────────

def upsert_referee_tendencies(records: list):
    """
    Upsert referee tendency records.
    Each record: {id, sport, referee, pace_factor, foul_rate, run_factor, games_officiated}.
    """
    if not records:
        return
    sb = _sb()
    for i in range(0, len(records), 500):
        chunk = _clean(records[i:i + 500])
        try:
            sb.table("referee_tendencies").upsert(chunk, on_conflict="id").execute()
        except Exception as exc:
            print(f"upsert_referee_tendencies error: {exc}")


def get_referee_tendency(sport: str, referee: str) -> dict:
    """Look up a referee's tendency stats."""
    ref_id = f"{sport}_{referee}".replace(" ", "_")[:200]
    try:
        res = (_sb().table("referee_tendencies")
               .select("*").eq("id", ref_id).execute())
        if res.data:
            return res.data[0]
    except Exception:
        pass
    return {}


# ── Pitcher-batter matchups ────────────────────────────────────────────────────

def upsert_pitcher_matchups(records: list):
    """
    Upsert pitcher-batter matchup stats.
    Each record: {id, pitcher, batter, ab, hits, k, bb, hr, matchup_win_rate}.
    """
    if not records:
        return
    sb = _sb()
    for i in range(0, len(records), 500):
        chunk = _clean(records[i:i + 500])
        try:
            sb.table("pitcher_matchup_stats").upsert(chunk, on_conflict="id").execute()
        except Exception as exc:
            print(f"upsert_pitcher_matchups error chunk {i}: {exc}")


def get_pitcher_batter_matchup_rate(pitcher: str, batter: str) -> float:
    """Historical matchup win rate for a specific pitcher vs batter."""
    mid = f"{pitcher}_{batter}".replace(" ", "_")[:200]
    try:
        res = (_sb().table("pitcher_matchup_stats")
               .select("matchup_win_rate").eq("id", mid).execute())
        if res.data:
            return float(res.data[0]["matchup_win_rate"])
    except Exception:
        pass
    return 0.5


# ── Weather cache ──────────────────────────────────────────────────────────────

def upsert_weather_cache(records: list):
    """
    Cache weather per venue per date to avoid repeated Open-Meteo calls.
    Each record: {id, venue, game_date, temp_max, precip_sum, windspeed_max}.
    """
    if not records:
        return
    sb = _sb()
    for i in range(0, len(records), 500):
        chunk = _clean(records[i:i + 500])
        try:
            sb.table("weather_cache").upsert(chunk, on_conflict="id").execute()
        except Exception as exc:
            print(f"upsert_weather_cache error: {exc}")


def get_cached_weather(venue: str, game_date: str) -> Optional[dict]:
    """Look up cached weather data."""
    wid = f"{venue}_{game_date}"
    try:
        res = (_sb().table("weather_cache")
               .select("*").eq("id", wid).execute())
        if res.data:
            return res.data[0]
    except Exception:
        pass
    return None
