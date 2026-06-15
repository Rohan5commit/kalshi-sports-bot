"""
data/scrapers/rotowire.py — Parses Rotowire RSS feeds for real-time injury
and lineup updates. No API key required.
"""
import time
import traceback
import xml.etree.ElementTree as ET

import requests

from config import MAX_RETRIES, BACKOFF_BASE

RSS_URLS = {
    "NBA": "https://www.rotowire.com/rss/news.php?sport=NBA",
    "NFL": "https://www.rotowire.com/rss/news.php?sport=NFL",
    "MLB": "https://www.rotowire.com/rss/news.php?sport=MLB",
}

INJURY_TERMS = {
    "out", "injured", "doubtful", "questionable", "day-to-day", "il",
    "inactive", "ruled out", "dnp", "won't play", "hamstring", "knee",
    "ankle", "shoulder", "concussion", "wrist", "hip", "back",
}

LINEUP_TERMS = {
    "starting", "will start", "confirmed", "set to start",
    "penciled in", "batting", "starting lineup", "lineup is set",
}


def _fetch(url: str) -> list:
    headers = {"User-Agent": "Mozilla/5.0"}
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            return [
                {
                    "title": item.findtext("title", ""),
                    "description": item.findtext("description", ""),
                    "published": item.findtext("pubDate", ""),
                }
                for item in root.findall(".//item")
            ]
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** attempt)
    return []


def _text(item: dict) -> str:
    return (item["title"] + " " + item["description"]).lower()


def _team_match(text: str, team_name: str) -> bool:
    return any(w in text for w in [w.lower() for w in team_name.split() if len(w) > 3])


def get_injury_updates(sport: str) -> list:
    items = _fetch(RSS_URLS.get(sport, ""))
    return [i for i in items if any(t in _text(i) for t in INJURY_TERMS)]


def get_lineup_updates(sport: str) -> list:
    items = _fetch(RSS_URLS.get(sport, ""))
    return [i for i in items if any(t in _text(i) for t in LINEUP_TERMS)]


def get_injury_flags(sport: str, home_team: str, away_team: str) -> dict:
    """Return {home_injury_flag, away_injury_flag} from Rotowire RSS."""
    items = get_injury_updates(sport)
    home_flag, away_flag = 0, 0
    for item in items:
        text = _text(item)
        if _team_match(text, home_team):
            home_flag = 1
        if _team_match(text, away_team):
            away_flag = 1
    return {"home_injury_flag": home_flag, "away_injury_flag": away_flag}


def get_lineup_confirmation(sport: str, home_team: str, away_team: str) -> dict:
    """Return {home_lineup_confirmed, away_lineup_confirmed} from Rotowire RSS."""
    items = get_lineup_updates(sport)
    home_conf, away_conf = 0, 0
    for item in items:
        text = _text(item)
        if _team_match(text, home_team):
            home_conf = 1
        if _team_match(text, away_team):
            away_conf = 1
    return {"home_lineup_confirmed": home_conf, "away_lineup_confirmed": away_conf}
