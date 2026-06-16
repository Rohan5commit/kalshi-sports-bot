"""
db/historical_store.py — Supabase storage for historical game logs, Elo ratings,
lineup stats, referee tendencies, pitcher-batter matchups, weather cache,
Kalshi/Polymarket market data, and data validation audit log.
"""
import os
from datetime import date, datetime
from typing import Optional

from supabase import create_client, Client

_GAME_TABLES = {"NBA": "nba_game_logs", "NFL": "nfl_game_logs", "MLB": "mlb_game_logs"}

_ESPN_CACHE: dict = {}


def _espn_fallback_games(sport: str, game_date: str) -> list:
    """Fetch games from ESPN for a date not yet in the DB (e.g. current season)."""
    from data.scrapers.espn import get_scoreboard
    from datetime import date as date_cls
    key = (sport, game_date)
    if key in _ESPN_CACHE:
        return _ESPN_CACHE[key]
    try:
        dt = date_cls.fromisoformat(game_date)
        games = get_scoreboard(sport, dt)
        rows = [
            {"game_id": g["id"], "game_date": game_date,
             "home_team": g["home_team"], "away_team": g["away_team"],
             "game_start_time": g.get("date", "")}
            for g in games if g.get("id") and g.get("home_team") and g.get("away_team")
        ]
        _ESPN_CACHE[key] = rows
        return rows
    except Exception as exc:
        print(f"ESPN fallback error {sport} {game_date}: {exc}")
        _ESPN_CACHE[key] = []
        return []


def _sb() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def _clean(rows: list) -> list:
    return [{k: v for k, v in row.items() if v is not None} for row in rows]


# ── Game logs ──────────────────────────────────────────────────────────────────

def get_last_game_date(sport: str) -> Optional[date]:
    try:
        res = (_sb().table(_GAME_TABLES[sport])
               .select("game_date").order("game_date", desc=True).limit(1).execute())
        if res.data:
            return datetime.strptime(res.data[0]["game_date"], "%Y-%m-%d").date()
    except Exception:
        pass
    return None


def upsert_game_logs(sport: str, logs: list):
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
    try:
        q = _sb().table(_GAME_TABLES[sport]).select("*").order("game_date").limit(limit)
        if since_date:
            q = q.gte("game_date", str(since_date))
        return q.execute().data
    except Exception as exc:
        print(f"get_game_logs error: {exc}")
        return []


def find_game_by_teams_and_date(sport: str, team1: str, team2: str,
                                 game_date: str) -> Optional[dict]:
    """Find a game record matching two teams on a specific date (fuzzy team name match)."""
    from rapidfuzz import fuzz
    try:
        table = _GAME_TABLES.get(sport)
        if not table:
            return None
        rows = (_sb().table(table)
                .select("game_id,game_date,home_team,away_team")
                .eq("game_date", game_date).execute().data)
        print(f"[DEBUG DB] sport={sport} date={game_date} rows={len(rows)}")
        if not rows:
            print(f"[DEBUG ESPN] calling fallback for {sport} {game_date}")
            rows = _espn_fallback_games(sport, game_date)
            print(f"[DEBUG ESPN] got {len(rows)} rows")
        best, best_score = None, 0.0
        for row in rows:
            home = row.get("home_team", "")
            away = row.get("away_team", "")
            score = (max(fuzz.WRatio(team1, home), fuzz.WRatio(team2, home)) +
                     max(fuzz.WRatio(team1, away), fuzz.WRatio(team2, away))) / 200.0
            if score > best_score:
                best_score = score
                best = row
        return best if best_score >= 0.70 else None
    except Exception as exc:
        print(f"find_game_by_teams_and_date error: {exc}")
        return None


# ── Elo ratings ────────────────────────────────────────────────────────────────

def upsert_elo_ratings(sport: str, ratings: dict):
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
    try:
        res = _sb().table("elo_ratings").select("team,elo").eq("sport", sport).execute()
        return {r["team"]: r["elo"] for r in res.data}
    except Exception:
        return {}


# ── NBA lineup stats ───────────────────────────────────────────────────────────

def upsert_lineup_stats(sport: str, records: list):
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
    ref_id = f"{sport}_{referee}".replace(" ", "_")[:200]
    try:
        res = (_sb().table("referee_tendencies").select("*").eq("id", ref_id).execute())
        if res.data:
            return res.data[0]
    except Exception:
        pass
    return {}


# ── Pitcher-batter matchups ────────────────────────────────────────────────────

def upsert_pitcher_matchups(records: list):
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
    wid = f"{venue}_{game_date}"
    try:
        res = (_sb().table("weather_cache").select("*").eq("id", wid).execute())
        if res.data:
            return res.data[0]
    except Exception:
        pass
    return None


# ── Kalshi market data ─────────────────────────────────────────────────────────

def upsert_kalshi_price_history(records: list):
    if not records:
        return
    sb = _sb()
    for i in range(0, len(records), 500):
        chunk = _clean(records[i:i + 500])
        try:
            sb.table("kalshi_price_history").upsert(chunk).execute()
        except Exception as exc:
            print(f"upsert_kalshi_price_history error chunk {i}: {exc}")


def upsert_kalshi_trades(records: list):
    if not records:
        return
    sb = _sb()
    for i in range(0, len(records), 500):
        chunk = _clean(records[i:i + 500])
        try:
            sb.table("kalshi_trades").upsert(chunk, on_conflict="id").execute()
        except Exception as exc:
            print(f"upsert_kalshi_trades error chunk {i}: {exc}")


def upsert_kalshi_game_matches(records: list):
    if not records:
        return
    sb = _sb()
    for i in range(0, len(records), 500):
        chunk = _clean(records[i:i + 500])
        try:
            sb.table("kalshi_game_matches").upsert(chunk, on_conflict="kalshi_ticker").execute()
        except Exception as exc:
            print(f"upsert_kalshi_game_matches error: {exc}")


# ── Polymarket data ────────────────────────────────────────────────────────────

def upsert_polymarket_price_history(records: list):
    if not records:
        return
    sb = _sb()
    for i in range(0, len(records), 500):
        chunk = _clean(records[i:i + 500])
        try:
            sb.table("polymarket_price_history").upsert(chunk).execute()
        except Exception as exc:
            print(f"upsert_polymarket_price_history error chunk {i}: {exc}")


def upsert_polymarket_trades(records: list):
    if not records:
        return
    sb = _sb()
    for i in range(0, len(records), 500):
        chunk = _clean(records[i:i + 500])
        try:
            sb.table("polymarket_trades").upsert(chunk, on_conflict="id").execute()
        except Exception as exc:
            print(f"upsert_polymarket_trades error chunk {i}: {exc}")


def upsert_polymarket_game_matches(records: list):
    if not records:
        return
    sb = _sb()
    for i in range(0, len(records), 500):
        chunk = _clean(records[i:i + 500])
        try:
            sb.table("polymarket_game_matches").upsert(chunk, on_conflict="market_id").execute()
        except Exception as exc:
            print(f"upsert_polymarket_game_matches error: {exc}")


def get_market_features_for_sport(sport: str) -> dict:
    """Return {espn_game_id: market_feature_dict} for enriching game log records."""
    result: dict = {}
    sb = _sb()

    try:
        rows = (sb.table("kalshi_game_matches")
                  .select("*").eq("sport", sport).execute().data)
        for r in rows:
            gid = r.get("espn_game_id", "")
            if not gid:
                continue
            if gid not in result:
                result[gid] = {}
            result[gid].update({
                "kalshi_open_price": r.get("open_price") or 0.5,
                "kalshi_close_price": r.get("close_price") or 0.5,
                "kalshi_price_movement": r.get("price_movement") or 0.0,
                "kalshi_total_volume": r.get("total_volume") or 0.0,
                "kalshi_last_hour_volume": r.get("last_hour_volume") or 0.0,
                "kalshi_price_at_tipoff": r.get("price_at_tipoff") or 0.5,
                "kalshi_max_single_move": r.get("max_single_move") or 0.0,
                "kalshi_days_open": r.get("days_open") or 0,
                "kalshi_overround": r.get("overround") or 0.0,
            })
    except Exception as exc:
        print(f"get_market_features (kalshi/{sport}) error: {exc}")

    try:
        rows = (sb.table("polymarket_game_matches")
                  .select("*").eq("sport", sport).execute().data)
        for r in rows:
            gid = r.get("espn_game_id", "")
            if not gid:
                continue
            if gid not in result:
                result[gid] = {}
            prev = result[gid]
            k_vol = float(prev.get("kalshi_total_volume") or 0)
            p_vol = float(r.get("total_volume") or 0)
            k_close = float(prev.get("kalshi_close_price") or 0.5)
            p_close = float(r.get("close_price") or 0.5)
            result[gid].update({
                "poly_open_price": r.get("open_price") or 0.5,
                "poly_close_price": p_close,
                "poly_price_movement": r.get("price_movement") or 0.0,
                "poly_total_volume": p_vol,
                "poly_last_hour_volume": r.get("last_hour_volume") or 0.0,
                "poly_price_at_tipoff": r.get("price_at_tipoff") or 0.5,
                "poly_max_single_move": r.get("max_single_move") or 0.0,
                "poly_days_open": r.get("days_open") or 0,
                "kalshi_vs_poly_spread": k_close - p_close,
                "kalshi_vs_poly_volume_ratio": (k_vol / (p_vol + 1e-6)
                                                if p_vol > 0 else 0.0),
            })
    except Exception as exc:
        print(f"get_market_features (polymarket/{sport}) error: {exc}")

    return result


# ── Data quality audit ─────────────────────────────────────────────────────────

def log_dropped_rows(records: list):
    if not records:
        return
    sb = _sb()
    for i in range(0, len(records), 500):
        chunk = records[i:i + 500]
        try:
            sb.table("dropped_rows").insert(chunk).execute()
        except Exception as exc:
            print(f"log_dropped_rows error chunk {i}: {exc}")

