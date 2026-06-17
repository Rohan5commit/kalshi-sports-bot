"""
data/scrapers/nba_data.py — Pull NBA game logs via basketball_reference_web_scraper.
basketball-reference.com is accessible from cloud IPs (stats.nba.com blocks Modal AWS).
"""
import time
import hashlib
from datetime import datetime, date

_BR_ABBREV = {
    "ATLANTA HAWKS": "ATL", "BOSTON CELTICS": "BOS", "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA", "CHICAGO BULLS": "CHI", "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL", "DENVER NUGGETS": "DEN", "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW", "HOUSTON ROCKETS": "HOU", "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC", "LOS ANGELES LAKERS": "LAL", "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA", "MILWAUKEE BUCKS": "MIL", "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP", "NEW YORK KNICKS": "NYK", "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL", "PHILADELPHIA 76ERS": "PHI", "PHOENIX SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR", "SACRAMENTO KINGS": "SAC", "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR", "UTAH JAZZ": "UTA", "WASHINGTON WIZARDS": "WAS",
    "NEW JERSEY NETS": "NJN", "CHARLOTTE BOBCATS": "CHA", "NEW ORLEANS HORNETS": "NOH",
    "SEATTLE SUPERSONICS": "SEA", "VANCOUVER GRIZZLIES": "MEM",
}


def _team_abbrev(team) -> str:
    name = team.value if hasattr(team, "value") else str(team)
    return _BR_ABBREV.get(name.upper(), name[:3].upper())


def _game_id(game_date: str, home: str, away: str) -> str:
    key = f"{game_date}_{home}_{away}"
    return "br_" + hashlib.md5(key.encode()).hexdigest()[:12]


def _get_season(season_end_year: int, retries: int = 2, backoff: int = 60) -> list:
    """Fetch season schedule from basketball-reference with retry on 429."""
    from basketball_reference_web_scraper import client
    for attempt in range(retries):
        try:
            time.sleep(3.0)
            return client.season_schedule(season_end_year=season_end_year)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "Too Many Requests" in msg:
                if attempt < retries - 1:
                    print(f"[NBA] BR {season_end_year}: rate limited, sleeping {backoff}s before retry...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    print(f"[NBA] BR {season_end_year}: rate limited after {retries} attempts, skipping")
                    return []
            else:
                print(f"[NBA] BR {season_end_year}: {exc}")
                return []
    return []


def _games_to_records(games: list) -> list:
    records = []
    for g in games:
        home_score = g.get("home_team_score")
        away_score = g.get("away_team_score")
        if home_score is None or away_score is None:
            continue

        game_date_raw = g.get("date") or g.get("start_time")
        if isinstance(game_date_raw, (datetime, date)):
            game_date = game_date_raw.strftime("%Y-%m-%d")
        elif isinstance(game_date_raw, str):
            game_date = game_date_raw[:10]
        else:
            continue

        home = _team_abbrev(g["home_team"])
        away = _team_abbrev(g["away_team"])

        records.append({
            "game_id": _game_id(game_date, home, away),
            "game_date": game_date,
            "home_team": home,
            "away_team": away,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "home_win": int(home_score) > int(away_score),
        })
    return records


def ingest_full_history(seasons: list, incremental_from=None) -> list:
    """Pull NBA game logs from basketball-reference.com for given NBA start seasons."""
    all_logs: list = []
    for season in seasons:
        season_end_year = season + 1  # e.g. 2022 start → 2023 end year
        games = _get_season(season_end_year)
        records = _games_to_records(games)
        if incremental_from:
            records = [g for g in records if g.get("game_date", "") > str(incremental_from)]
        all_logs.extend(records)
        print(f"[NBA] BR {season} ({season}-{str(season_end_year)[-2:]}): {len(records)} games")
        time.sleep(5.0)  # extra cooldown between seasons to avoid BR rate limiting
    return all_logs
