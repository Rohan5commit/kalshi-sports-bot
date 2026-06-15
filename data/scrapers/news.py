"""
data/scrapers/news.py — Currents API scraper for injury reports and lineup news.
Requires CURRENTS_API_KEY in Modal Secret (newsapi-secret).
"""
import os
import time
import traceback
import requests
from datetime import datetime, timedelta

from config import CURRENTS_API_BASE, MAX_RETRIES, BACKOFF_BASE

TEAM_INJURY_SIGNAL = {
    "NBA": ["injury", "injured", "out", "doubtful", "questionable"],
    "NFL": ["injury", "injured", "out", "doubtful", "questionable", "ruled out"],
    "MLB": ["injury", "IL", "disabled list", "out"],
}

SPORT_QUERY_MAP = {
    "NBA": "NBA injury lineup",
    "NFL": "NFL injury lineup",
    "MLB": "MLB injury lineup",
}


def _search(keywords: str) -> list:
    api_key = os.environ["CURRENTS_API_KEY"]
    url = f"{CURRENTS_API_BASE}/search"
    params = {"keywords": keywords, "apiKey": api_key, "language": "en"}
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json().get("news", [])
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** attempt)
    raise last_exc


def get_injury_news(sport: str, lookback_days: int = 1) -> list:
    """
    Fetch recent injury/lineup news for a sport via Currents API.
    Returns list of dicts: {title, description, published_at, is_injury, source}.
    """
    try:
        articles = _search(SPORT_QUERY_MAP.get(sport, sport))
        keywords = TEAM_INJURY_SIGNAL.get(sport, [])
        results = []
        for art in articles:
            title = art.get("title", "") or ""
            desc = art.get("description", "") or ""
            text = (title + " " + desc).lower()
            is_injury = any(kw.lower() in text for kw in keywords)
            results.append({
                "title": title,
                "description": desc,
                "published_at": art.get("published", ""),
                "is_injury": is_injury,
                "source": art.get("author", ""),
            })
        return results
    except Exception as exc:
        try:
            from db.supabase_client import log_error
            log_error(
                context=f"news.get_injury_news({sport})",
                error_msg=str(exc),
                tb=traceback.format_exc(),
            )
        except Exception:
            pass
        return []


def build_injury_flags(sport: str, team_names: list) -> dict:
    articles = get_injury_news(sport, lookback_days=2)
    flags = {team: 0 for team in team_names}
    for art in articles:
        if not art["is_injury"]:
            continue
        text = (art["title"] + " " + art["description"]).lower()
        for team in team_names:
            words = [w.lower() for w in team.split() if len(w) > 3]
            if any(w in text for w in words):
                flags[team] = 1
    return flags


def get_all_injury_flags(sport: str, home_team: str, away_team: str) -> dict:
    flags = build_injury_flags(sport, [home_team, away_team])
    return {
        "home_injury_flag": flags.get(home_team, 0),
        "away_injury_flag": flags.get(away_team, 0),
    }
