"""
data/scrapers/sports_ref.py — Historical stats for model training.
Primary: sportsreference Python package (free, no key).
Fallback: ESPN scoreboard API pulling past seasons via date ranges.
Sports Reference often blocks cloud IPs, so ESPN fallback is critical.
"""
import time
import traceback
from datetime import date, timedelta
from typing import Optional

import requests

from config import HISTORICAL_SEASONS, ESPN_BASE, ESPN_SPORT_PATHS, MAX_RETRIES, BACKOFF_BASE
from db.supabase_client import log_error


def _retry_call(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** attempt)
    raise last_exc


def _espn_get(url: str, params: dict = None) -> dict:
    """ESPN API GET with retry."""
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


# ── ESPN historical fallback ──────────────────────────────────────────────────

def _espn_season_logs(sport: str, season_year: int) -> list:
    """
    Pull all completed games for a sport/season from ESPN.
    season_year is the year the season ends (e.g., 2025 for 2024-25 NBA).
    Returns list of game dicts compatible with build_training_dataset().
    """
    sport_path = ESPN_SPORT_PATHS.get(sport)
    if not sport_path:
        return []

    url = f"{ESPN_BASE}/{sport_path}/scoreboard"
    records = []

    # For basketball/football, build week-by-week or monthly date windows
    # ESPN scoreboard accepts a date range via 'dates' param (YYYYMMDD format)
    # We'll sample one date per week across the season

    if sport == "NBA":
        # NBA regular season: Oct → Apr
        start = date(season_year - 1, 10, 1)
        end = date(season_year, 4, 30)
    elif sport == "NFL":
        # NFL season: Sep → Feb
        start = date(season_year - 1, 9, 1)
        end = date(season_year, 2, 15)
    elif sport == "MLB":
        # MLB regular season: Apr → Oct
        start = date(season_year - 1, 4, 1)
        end = date(season_year - 1, 10, 15)
    else:
        return []

    current = start
    while current <= end:
        try:
            data = _espn_get(url, params={"dates": current.strftime("%Y%m%d"), "limit": 100})
            for event in data.get("events", []):
                try:
                    competition = event.get("competitions", [{}])[0]
                    status = competition.get("status", {}).get("type", {})
                    if not status.get("completed", False):
                        current += timedelta(days=7)
                        continue

                    competitors = competition.get("competitors", [])
                    home = next((c for c in competitors if c.get("homeAway") == "home"), {})
                    away = next((c for c in competitors if c.get("homeAway") == "away"), {})

                    home_score = int(home.get("score", 0) or 0)
                    away_score = int(away.get("score", 0) or 0)
                    home_win = home.get("winner", False)

                    record = {
                        "sport": sport,
                        "season": season_year,
                        "team": home.get("team", {}).get("abbreviation", ""),
                        "opponent": away.get("team", {}).get("abbreviation", ""),
                        "date": event.get("date", ""),
                        "home": True,
                        "win": home_win,
                    }

                    if sport == "NBA":
                        record.update({
                            "points": home_score,
                            "opp_points": away_score,
                            "fg_pct": 0.0, "fg3_pct": 0.0, "ft_pct": 0.0,
                            "rebounds": 0.0, "assists": 0.0, "turnovers": 0.0,
                            "steals": 0.0, "blocks": 0.0,
                        })
                    elif sport == "NFL":
                        record.update({
                            "points": home_score,
                            "opp_points": away_score,
                            "pass_yards": 0.0, "rush_yards": 0.0,
                            "turnovers": 0.0, "penalties": 0.0,
                            "first_downs": 0.0, "time_of_possession": 0.0,
                        })
                    elif sport == "MLB":
                        record.update({
                            "runs": home_score,
                            "opp_runs": away_score,
                            "hits": 0.0, "errors": 0.0,
                            "batting_avg": 0.0, "era": 0.0,
                            "strikeouts": 0.0, "walks": 0.0,
                        })

                    records.append(record)
                    # Also add away team perspective
                    away_record = {**record, "team": record["opponent"],
                                   "opponent": record["team"], "home": False, "win": not home_win}
                    if sport == "NBA":
                        away_record["points"] = away_score
                        away_record["opp_points"] = home_score
                    elif sport == "NFL":
                        away_record["points"] = away_score
                        away_record["opp_points"] = home_score
                    elif sport == "MLB":
                        away_record["runs"] = away_score
                        away_record["opp_runs"] = home_score
                    records.append(away_record)
                except Exception:
                    continue
        except Exception:
            pass
        current += timedelta(days=7)

    return records


def _espn_historical_logs(sport: str, seasons: int = HISTORICAL_SEASONS) -> list:
    """Pull multiple seasons of historical game data from ESPN."""
    import datetime
    current_year = datetime.date.today().year
    # Use completed seasons only
    season_years = list(range(current_year - seasons, current_year))

    all_records = []
    for year in season_years:
        try:
            print(f"  ESPN: fetching {sport} season {year}...")
            records = _espn_season_logs(sport, year)
            print(f"  ESPN: got {len(records)} records for {sport} {year}")
            all_records.extend(records)
        except Exception as exc:
            log_error(
                context=f"sports_ref._espn_historical_logs({sport}, {year})",
                error_msg=str(exc),
                tb=traceback.format_exc(),
            )
    return all_records


# ── sportsreference primary (with ESPN fallback) ──────────────────────────────

def get_nba_game_logs(seasons: int = HISTORICAL_SEASONS) -> list:
    """Pull NBA game logs. Tries sportsreference first, falls back to ESPN."""
    try:
        from sportsreference.nba.schedule import Schedule
        from sportsreference.nba.teams import Teams
        import datetime

        records = []
        current_year = datetime.date.today().year
        season_years = range(current_year - seasons, current_year)

        teams_obj = _retry_call(Teams)
        team_abbrevs = [t.abbreviation for t in teams_obj]

        for year in season_years:
            for abbrev in team_abbrevs[:5]:  # test with first 5 teams
                try:
                    schedule = _retry_call(Schedule, abbrev, year=year)
                    for game in schedule:
                        try:
                            records.append({
                                "sport": "NBA", "season": year,
                                "team": abbrev, "opponent": game.opponent_abbr,
                                "date": str(game.date), "home": game.location == "Home",
                                "win": game.result == "Win",
                                "points": game.points, "opp_points": game.opponent_points,
                                "fg_pct": game.field_goal_percentage or 0.0,
                                "fg3_pct": game.three_point_field_goal_percentage or 0.0,
                                "ft_pct": game.free_throw_percentage or 0.0,
                                "rebounds": game.total_rebounds or 0.0,
                                "assists": game.assists or 0.0,
                                "turnovers": game.turnovers or 0.0,
                                "steals": game.steals or 0.0,
                                "blocks": game.blocks or 0.0,
                            })
                        except Exception:
                            continue
                except Exception:
                    continue

        if records:
            print(f"sportsreference: got {len(records)} NBA records")
            return records
    except Exception as exc:
        print(f"sportsreference NBA failed ({exc}), falling back to ESPN...")

    return _espn_historical_logs("NBA", seasons)


def get_nfl_game_logs(seasons: int = HISTORICAL_SEASONS) -> list:
    """Pull NFL game logs. Tries sportsreference first, falls back to ESPN."""
    try:
        from sportsreference.nfl.schedule import Schedule
        from sportsreference.nfl.teams import Teams
        import datetime

        records = []
        current_year = datetime.date.today().year
        season_years = range(current_year - seasons, current_year)

        teams_obj = _retry_call(Teams)
        team_abbrevs = [t.abbreviation for t in teams_obj]

        for year in season_years:
            for abbrev in team_abbrevs[:5]:
                try:
                    schedule = _retry_call(Schedule, abbrev, year=year)
                    for game in schedule:
                        try:
                            records.append({
                                "sport": "NFL", "season": year,
                                "team": abbrev, "opponent": game.opponent_abbr,
                                "date": str(game.date), "home": game.location == "Home",
                                "win": game.result == "Win",
                                "points": game.points, "opp_points": game.opponent_points,
                                "pass_yards": game.pass_yards or 0.0,
                                "rush_yards": game.rush_yards or 0.0,
                                "turnovers": game.turnovers or 0.0,
                                "penalties": game.penalties or 0.0,
                                "first_downs": game.first_downs or 0.0,
                                "time_of_possession": game.time_of_possession or 0.0,
                            })
                        except Exception:
                            continue
                except Exception:
                    continue

        if records:
            print(f"sportsreference: got {len(records)} NFL records")
            return records
    except Exception as exc:
        print(f"sportsreference NFL failed ({exc}), falling back to ESPN...")

    return _espn_historical_logs("NFL", seasons)


def get_mlb_game_logs(seasons: int = HISTORICAL_SEASONS) -> list:
    """Pull MLB game logs. Tries sportsreference first, falls back to ESPN."""
    try:
        from sportsreference.mlb.schedule import Schedule
        from sportsreference.mlb.teams import Teams
        import datetime

        records = []
        current_year = datetime.date.today().year
        season_years = range(current_year - seasons, current_year)

        teams_obj = _retry_call(Teams)
        team_abbrevs = [t.abbreviation for t in teams_obj]

        for year in season_years:
            for abbrev in team_abbrevs[:5]:
                try:
                    schedule = _retry_call(Schedule, abbrev, year=year)
                    for game in schedule:
                        try:
                            records.append({
                                "sport": "MLB", "season": year,
                                "team": abbrev, "opponent": game.opponent_abbr,
                                "date": str(game.date), "home": game.location == "Home",
                                "win": game.result == "Win",
                                "runs": game.runs, "opp_runs": game.opponent_runs,
                                "hits": game.hits or 0.0,
                                "errors": game.errors or 0.0,
                                "batting_avg": game.batting_average or 0.0,
                                "era": game.earned_run_average or 0.0,
                                "strikeouts": game.strikeouts or 0.0,
                                "walks": game.walks or 0.0,
                            })
                        except Exception:
                            continue
                except Exception:
                    continue

        if records:
            print(f"sportsreference: got {len(records)} MLB records")
            return records
    except Exception as exc:
        print(f"sportsreference MLB failed ({exc}), falling back to ESPN...")

    return _espn_historical_logs("MLB", seasons)


def get_all_historical_data() -> dict:
    """Pull all historical game logs for NBA, NFL, MLB."""
    return {
        "NBA": get_nba_game_logs(),
        "NFL": get_nfl_game_logs(),
        "MLB": get_mlb_game_logs(),
    }
