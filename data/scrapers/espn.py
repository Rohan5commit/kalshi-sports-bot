"""
data/scrapers/espn.py — ESPN hidden API scraper for schedules, scores, and team stats.
No auth required; uses site.api.espn.com.
"""
import time
import traceback
import requests
from datetime import date, datetime
from typing import Optional

from config import ESPN_BASE, ESPN_SPORT_PATHS, MAX_RETRIES, BACKOFF_BASE
from db.supabase_client import log_error


def _get(url: str, params: dict = None) -> dict:
    """GET with retry + exponential backoff."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** attempt)
    raise last_exc


def _american_to_prob(odds_str: str) -> Optional[float]:
    """Convert American moneyline string (e.g. '-118', '+130') to raw implied probability."""
    try:
        o = float(str(odds_str).replace("+", ""))
        if o < 0:
            return (-o) / (-o + 100)
        else:
            return 100 / (o + 100)
    except (ValueError, TypeError):
        return None


def _extract_moneyline(competition: dict) -> Optional[float]:
    """
    Extract vig-removed home win probability from ESPN DraftKings moneyline.
    Returns float (0-1) or None if odds unavailable.
    """
    odds_list = competition.get("odds", [])
    for odds_entry in odds_list:
        ml = odds_entry.get("moneyline", {})
        home_odds_str = (ml.get("home", {}).get("close", {}) or ml.get("home", {}).get("open", {})).get("odds")
        away_odds_str = (ml.get("away", {}).get("close", {}) or ml.get("away", {}).get("open", {})).get("odds")
        if home_odds_str and away_odds_str:
            home_raw = _american_to_prob(home_odds_str)
            away_raw = _american_to_prob(away_odds_str)
            if home_raw and away_raw:
                total = home_raw + away_raw
                return round(home_raw / total, 4)
    return None


def get_scoreboard(sport: str, game_date: Optional[date] = None) -> list:
    """
    Fetch today's (or a specific date's) scoreboard for sport.
    Returns list of game dicts including vegas_home_prob from DraftKings moneylines.
    """
    sport_path = ESPN_SPORT_PATHS.get(sport)
    if not sport_path:
        raise ValueError(f"Unknown sport: {sport}")

    url = f"{ESPN_BASE}/{sport_path}/scoreboard"
    params = {}
    if game_date:
        params["dates"] = game_date.strftime("%Y%m%d")

    try:
        data = _get(url, params)
        games = []
        for event in data.get("events", []):
            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})

            status_type = event.get("status", {}).get("type", {})
            vegas_prob = _extract_moneyline(competition)
            games.append({
                "id": event.get("id"),
                "name": event.get("name", ""),
                "sport": sport,
                "date": event.get("date", ""),
                "status": status_type.get("name", ""),
                "completed": status_type.get("completed", False),
                "home_team": home.get("team", {}).get("displayName", ""),
                "home_abbr": home.get("team", {}).get("abbreviation", ""),
                "home_id": home.get("team", {}).get("id", ""),
                "home_score": int(home.get("score", 0) or 0),
                "away_team": away.get("team", {}).get("displayName", ""),
                "away_abbr": away.get("team", {}).get("abbreviation", ""),
                "away_id": away.get("team", {}).get("id", ""),
                "away_score": int(away.get("score", 0) or 0),
                "home_winner": home.get("winner", False),
                "vegas_home_prob": vegas_prob,
            })
        return games
    except Exception as exc:
        log_error(
            context=f"espn.get_scoreboard({sport})",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        raise


def get_team_stats(sport: str, team_id: str) -> dict:
    """
    Fetch season stats for a specific team.
    Returns a flat dict of stat name → value.
    """
    sport_path = ESPN_SPORT_PATHS.get(sport)
    if not sport_path:
        raise ValueError(f"Unknown sport: {sport}")

    url = f"{ESPN_BASE}/{sport_path}/teams/{team_id}"
    try:
        data = _get(url)
        team = data.get("team", {})
        record = team.get("record", {}).get("items", [{}])[0]
        stats_raw = record.get("stats", [])
        stats = {s["name"]: s.get("value", 0.0) for s in stats_raw}
        stats["team_id"] = team_id
        stats["team_name"] = team.get("displayName", "")
        return stats
    except Exception as exc:
        log_error(
            context=f"espn.get_team_stats({sport}, {team_id})",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return {}


def get_standings(sport: str) -> list:
    """
    Fetch current standings for sport.
    Returns list of dicts with team id, name, wins, losses.
    """
    sport_path = ESPN_SPORT_PATHS.get(sport)
    if not sport_path:
        raise ValueError(f"Unknown sport: {sport}")

    url = f"{ESPN_BASE}/{sport_path}/standings"
    try:
        data = _get(url)
        entries = []
        for group in data.get("children", []):
            for entry in group.get("standings", {}).get("entries", []):
                team = entry.get("team", {})
                stats = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}
                entries.append({
                    "team_id": team.get("id", ""),
                    "team_name": team.get("displayName", ""),
                    "wins": stats.get("wins", 0),
                    "losses": stats.get("losses", 0),
                    "win_pct": stats.get("winPercent", 0.0),
                    "points_for": stats.get("pointsFor", 0.0),
                    "points_against": stats.get("pointsAgainst", 0.0),
                    "streak": stats.get("streak", 0),
                })
        return entries
    except Exception as exc:
        log_error(
            context=f"espn.get_standings({sport})",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return []


def get_recent_results(sport: str, team_id: str, n: int = 10) -> list:
    """
    Fetch last n game results for a team to compute recent form.
    Returns list of dicts with win (bool), points_scored, points_allowed.
    """
    sport_path = ESPN_SPORT_PATHS.get(sport)
    if not sport_path:
        raise ValueError(f"Unknown sport: {sport}")

    url = f"{ESPN_BASE}/{sport_path}/teams/{team_id}/schedule"
    try:
        data = _get(url)
        events = data.get("events", [])
        results = []
        for event in events:
            competition = event.get("competitions", [{}])[0]
            if not competition.get("status", {}).get("type", {}).get("completed", False):
                continue
            competitors = competition.get("competitors", [])
            team_c = next((c for c in competitors if c.get("team", {}).get("id") == team_id), None)
            opp_c = next((c for c in competitors if c.get("team", {}).get("id") != team_id), None)
            if not team_c:
                continue
            results.append({
                "game_id": event.get("id"),
                "date": event.get("date"),
                "win": team_c.get("winner", False),
                "points_scored": int(team_c.get("score", 0) or 0),
                "points_allowed": int(opp_c.get("score", 0) or 0) if opp_c else 0,
                "home": team_c.get("homeAway") == "home",
            })
        return results[-n:]
    except Exception as exc:
        log_error(
            context=f"espn.get_recent_results({sport}, {team_id})",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return []


def get_head_to_head(sport: str, team_id: str, opponent_id: str, n: int = 5) -> dict:
    """
    Compute H2H record between two teams from recent schedule.
    Returns dict with wins, losses, avg_margin.
    """
    results = get_recent_results(sport, team_id, n=50)
    # ESPN schedule doesn't expose opponent ID easily, so we filter by score records
    # This is a simplified H2H based on available data
    h2h = {"wins": 0, "losses": 0, "avg_margin": 0.0}
    margins = []
    for r in results:
        margins.append(r["points_scored"] - r["points_allowed"])
        if r["win"]:
            h2h["wins"] += 1
        else:
            h2h["losses"] += 1
    h2h["avg_margin"] = sum(margins) / len(margins) if margins else 0.0
    return h2h
