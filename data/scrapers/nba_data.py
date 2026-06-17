"""
data/scrapers/nba_data.py — Pull NBA game logs by scraping basketball-reference.com
directly with browser headers (library-based calls get 403/429 blocked on cloud IPs).
"""
import time
import hashlib
import re
from datetime import datetime, date as _date

import pandas as pd

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

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.basketball-reference.com/",
    "Connection": "keep-alive",
}

_MONTHS = ["october", "november", "december", "january", "february",
           "march", "april", "may", "june"]


def _team_abbrev(name: str) -> str:
    return _BR_ABBREV.get(name.strip().upper(), name.strip()[:3].upper())


def _game_id(game_date: str, home: str, away: str) -> str:
    return "br_" + hashlib.md5(f"{game_date}_{home}_{away}".encode()).hexdigest()[:12]


def _parse_date(raw: str) -> str:
    """Convert 'Mon, Dec 25, 2025' or 'October 24, 2025' → 'YYYY-MM-DD'."""
    raw = re.sub(r"^[A-Za-z]+,\s*", "", raw.strip())
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw[:10]


def _fetch_month(season_end_year: int, month: str, session) -> list:
    url = (
        f"https://www.basketball-reference.com/leagues/"
        f"NBA_{season_end_year}_games-{month}.html"
    )
    try:
        time.sleep(2.5)
        resp = session.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code == 404:
            return []
        if resp.status_code == 429:
            print(f"[NBA] BR {season_end_year}-{month}: 429 rate limited, sleeping 90s")
            time.sleep(90)
            resp = session.get(url, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
        resp.raise_for_status()
    except Exception as exc:
        print(f"[NBA] BR {season_end_year}-{month}: {exc}")
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="schedule")
    if not table:
        return []

    records = []
    for row in table.find("tbody").find_all("tr"):
        if "thead" in (row.get("class") or []):
            continue
        date_th = row.find("th", {"data-stat": "date_game"})
        vis_td = row.find("td", {"data-stat": "visitor_team_name"})
        vis_pts = row.find("td", {"data-stat": "visitor_pts"})
        home_td = row.find("td", {"data-stat": "home_team_name"})
        home_pts = row.find("td", {"data-stat": "home_pts"})
        if not all([date_th, vis_td, home_td, vis_pts, home_pts]):
            continue
        pts_str = vis_pts.text.strip()
        if not pts_str or not pts_str.isdigit():
            continue  # game not played yet
        try:
            game_date = _parse_date(date_th.text)
            home = _team_abbrev(home_td.text)
            away = _team_abbrev(vis_td.text)
            h_pts = int(home_pts.text.strip())
            a_pts = int(pts_str)
            records.append({
                "game_id": _game_id(game_date, home, away),
                "game_date": game_date,
                "home_team": home,
                "away_team": away,
                "home_score": h_pts,
                "away_score": a_pts,
                "home_win": h_pts > a_pts,
            })
        except (ValueError, AttributeError):
            continue
    return records


def _get_season(season_end_year: int) -> list:
    import requests
    session = requests.Session()
    all_records = []
    for month in _MONTHS:
        recs = _fetch_month(season_end_year, month, session)
        all_records.extend(recs)
    return all_records


def ingest_full_history(seasons: list, incremental_from=None) -> list:
    """Pull NBA game logs from basketball-reference.com for given NBA start seasons.

    In incremental mode, skips seasons that ended in a prior calendar year to avoid
    exhausting rate limits on already-loaded data.
    """
    all_logs: list = []
    current_year = _date.today().year
    for season in seasons:
        season_end_year = season + 1
        if incremental_from and season_end_year < current_year:
            print(f"[NBA] BR {season} ({season}-{str(season_end_year)[-2:]}): already loaded, skipping")
            continue
        records = _get_season(season_end_year)
        if incremental_from:
            records = [g for g in records if g.get("game_date", "") > str(incremental_from)]
        all_logs.extend(records)
        print(f"[NBA] BR {season} ({season}-{str(season_end_year)[-2:]}): {len(records)} games")
        time.sleep(5.0)
    return all_logs
