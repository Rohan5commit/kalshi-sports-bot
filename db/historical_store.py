"""
db/historical_store.py — Supabase storage for historical game logs, Elo ratings,
lineup stats, referee tendencies, pitcher-batter matchups, weather cache,
Kalshi/Polymarket market data, and data validation audit log.
"""
import os
import threading
from datetime import date, datetime
from typing import Optional

from supabase import create_client, Client

_GAME_TABLES = {"NBA": "nba_game_logs", "NFL": "nfl_game_logs", "MLB": "mlb_game_logs"}

_NFL_NAME_TO_ABBREV = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Oakland Raiders": "LV",
    "Los Angeles Chargers": "LAC", "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LA", "St. Louis Rams": "LA",
    "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO",
    "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS", "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}

_MLB_NAME_TO_ABBREV = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY",
    "Oakland Athletics": "ATH", "Athletics": "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS",
}

_NBA_NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

_SPORT_ABBREV_MAP = {"NFL": _NFL_NAME_TO_ABBREV, "MLB": _MLB_NAME_TO_ABBREV, "NBA": _NBA_NAME_TO_ABBREV}

_SB_CLIENT = None
_SB_LOCK = threading.Lock()

def _sb() -> Client:
    global _SB_CLIENT
    if _SB_CLIENT is not None:
        return _SB_CLIENT
    with _SB_LOCK:
        if _SB_CLIENT is None:
            _SB_CLIENT = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    return _SB_CLIENT


def _clean(rows: list) -> list:
    return [{k: v for k, v in row.items() if v is not None} for row in rows]


# ── espn_schedule in-memory cache ─────────────────────────────────────────────

_ESPN_SCHEDULE_CACHE: dict = {}
_ESPN_CACHE_LOCK = threading.Lock()
_ESPN_CACHE_LOADED = False


def ensure_espn_cache():
    """Pre-load espn_schedule into memory. Call once in main thread before workers start."""
    global _ESPN_SCHEDULE_CACHE, _ESPN_CACHE_LOADED
    if _ESPN_CACHE_LOADED:
        return
    with _ESPN_CACHE_LOCK:
        if _ESPN_CACHE_LOADED:
            return
        print("[Cache] Loading espn_schedule into memory (paginated)...")
        cache: dict = {}
        total = 0
        try:
            offset, page_size = 0, 1000
            while True:
                rows = (_sb().table("espn_schedule")
                        .select("game_id,sport,game_date,home_team,away_team")
                        .range(offset, offset + page_size - 1)
                        .execute().data)
                if not rows:
                    break
                for row in rows:
                    key = (row.get("sport", ""), row.get("game_date", ""))
                    cache.setdefault(key, []).append(row)
                total += len(rows)
                if len(rows) < page_size:
                    break
                offset += page_size
            print(f"[Cache] {total} rows loaded, {len(cache)} date-buckets")
        except Exception as exc:
            print(f"[Cache] WARNING: load failed ({exc}) — using live queries as fallback")
        finally:
            _ESPN_SCHEDULE_CACHE = cache
            _ESPN_CACHE_LOADED = True  # always mark done to stop retry storm


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
                  limit: int = 50000) -> list:
    try:
        table = _GAME_TABLES[sport]
        all_rows: list = []
        page_size = 1000
        offset = 0
        while len(all_rows) < limit:
            q = (_sb().table(table).select("*")
                 .order("game_date").range(offset, offset + page_size - 1))
            if since_date:
                q = q.gte("game_date", str(since_date))
            rows = q.execute().data
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < page_size:
                break
            offset += page_size
        return all_rows
    except Exception as exc:
        print(f"get_game_logs error: {exc}")
        return []


def find_game_by_teams_and_date(sport: str, team1: str, team2: str,
                                 game_date: str) -> Optional[dict]:
    """Find a game record matching two teams on a specific date (fuzzy team name match)."""
    from rapidfuzz import fuzz
    try:
        # In-memory lookup — fast path (no Supabase round trip)
        rows = _ESPN_SCHEDULE_CACHE.get((sport, game_date), [])
        if not rows:
            # Cache miss or failed load — live fallback
            table = _GAME_TABLES.get(sport)
            if table:
                try:
                    rows = (_sb().table(table)
                            .select("game_id,game_date,home_team,away_team")
                            .eq("game_date", game_date).execute().data)
                except Exception:
                    pass
            if not rows:
                try:
                    rows = (_sb().table("espn_schedule")
                            .select("game_id,game_date,home_team,away_team")
                            .eq("sport", sport).eq("game_date", game_date).execute().data)
                except Exception:
                    pass
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
            print(f"upsert_referee_tendencies error chunk {i}: {exc}")


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


def _paginate_market_table(sb, table: str, sport: str) -> list:
    rows, offset, page_size = [], 0, 1000
    while True:
        batch = (sb.table(table).select("*").eq("sport", sport)
                 .range(offset, offset + page_size - 1).execute().data)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def get_market_features_for_sport(sport: str) -> dict:
    """Return features keyed by espn_game_id AND by 'game_date_homeabbrev' for cross-format matching."""
    result: dict = {}
    sb = _sb()
    abbrev_map = _SPORT_ABBREV_MAP.get(sport, {})

    def _add(r: dict, feats: dict):
        gid = r.get("espn_game_id", "")
        if gid:
            result.setdefault(gid, {}).update(feats)
        gdate = r.get("game_date", "")
        home = r.get("home_team", "")
        abbrev = abbrev_map.get(home, "")
        if gdate and abbrev:
            result.setdefault(f"{gdate}_{abbrev}", {}).update(feats)

    try:
        for r in _paginate_market_table(sb, "kalshi_game_matches", sport):
            _add(r, {
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
        for r in _paginate_market_table(sb, "polymarket_game_matches", sport):
            gid = r.get("espn_game_id", "")
            prev = result.get(gid, {})
            k_vol = float(prev.get("kalshi_total_volume") or 0)
            p_vol = float(r.get("total_volume") or 0)
            k_close = float(prev.get("kalshi_close_price") or 0.5)
            p_close = float(r.get("close_price") or 0.5)
            _add(r, {
                "poly_open_price": r.get("open_price") or 0.5,
                "poly_close_price": p_close,
                "poly_price_movement": r.get("price_movement") or 0.0,
                "poly_total_volume": p_vol,
                "poly_last_hour_volume": r.get("last_hour_volume") or 0.0,
                "poly_price_at_tipoff": r.get("price_at_tipoff") or 0.5,
                "poly_max_single_move": r.get("max_single_move") or 0.0,
                "poly_days_open": r.get("days_open") or 0,
                "kalshi_vs_poly_spread": k_close - p_close,
                "kalshi_vs_poly_volume_ratio": (k_vol / (p_vol + 1e-6) if p_vol > 0 else 0.0),
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
