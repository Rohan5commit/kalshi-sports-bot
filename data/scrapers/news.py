"""
data/scrapers/news.py — NewsAPI scraper for injury reports and lineup news.
Requires NEWS_API_KEY in Modal Secret.
"""
import os
import time
import traceback
import requests
from datetime import date, datetime, timedelta
from typing import Optional

from config import NEWS_API_BASE, MAX_RETRIES, BACKOFF_BASE, SPORTS
from db.supabase_client import log_error

INJURY_KEYWORDS = ["injury", "injured", "out", "questionable", "doubtful", "DNP",
                   "ruled out", "missed", "inactive", "knee", "ankle", "shoulder",
                   "hamstring", "concussion", "day-to-day"]


def _get(url: str, params: dict = None) -> dict:
    last_exc = None
    api_key = os.environ["NEWS_API_KEY"]
    headers = {"X-Api-Key": api_key}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** attempt)
    raise last_exc


SPORT_QUERY_MAP = {
    "NBA": "NBA injury OR NBA lineup",
    "NFL": "NFL injury OR NFL lineup",
    "MLB": "MLB injury OR MLB lineup",
}

TEAM_INJURY_SIGNAL = {
    "NBA": ["injury", "injured", "out", "doubtful", "questionable"],
    "NFL": ["injury", "injured", "out", "doubtful", "questionable", "ruled out"],
    "MLB": ["injury", "IL", "disabled list", "out"],
}


def get_injury_news(sport: str, lookback_days: int = 1) -> list:
    """
    Fetch recent injury / lineup news for a sport.
    Returns list of dicts: {title, team_hints, is_injury, published_at}.
    """
    if sport not in SPORT_QUERY_MAP:
        raise ValueError(f"Unknown sport: {sport}")

    from_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = f"{NEWS_API_BASE}/everything"
    params = {
        "q": SPORT_QUERY_MAP[sport],
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,
    }
    try:
        data = _get(url, params)
        articles = data.get("articles", [])
        results = []
        for art in articles:
            title = art.get("title", "") or ""
            desc = art.get("description", "") or ""
            text = (title + " " + desc).lower()
            keywords = TEAM_INJURY_SIGNAL.get(sport, [])
            is_injury = any(kw.lower() in text for kw in keywords)
            results.append({
                "title": title,
                "description": desc,
                "published_at": art.get("publishedAt", ""),
                "is_injury": is_injury,
                "source": art.get("source", {}).get("name", ""),
            })
        return results
    except Exception as exc:
        log_error(
            context=f"news.get_injury_news({sport})",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return []


def build_injury_flags(sport: str, team_names: list) -> dict:
    """
    For a list of team names, return a dict mapping team_name → injury_flag (0 or 1).
    1 means at least one injury article mentions that team.
    """
    articles = get_injury_news(sport, lookback_days=2)
    flags = {team: 0 for team in team_names}

    for art in articles:
        if not art["is_injury"]:
            continue
        text = (art["title"] + " " + art["description"]).lower()
        for team in team_names:
            # Match on any word in the team name
            words = [w.lower() for w in team.split() if len(w) > 3]
            if any(w in text for w in words):
                flags[team] = 1

    return flags


def get_all_injury_flags(sport: str, home_team: str, away_team: str) -> dict:
    """
    Convenience wrapper: return injury flags for exactly two teams.
    Keys: home_injury_flag, away_injury_flag.
    """
    flags = build_injury_flags(sport, [home_team, away_team])
    return {
        "home_injury_flag": flags.get(home_team, 0),
        "away_injury_flag": flags.get(away_team, 0),
    }
