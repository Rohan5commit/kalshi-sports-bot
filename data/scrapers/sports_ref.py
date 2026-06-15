"""
data/scrapers/sports_ref.py — Historical stats via the sportsreference Python package.
Pulls 3–5 seasons of game logs for training. No API key required.
"""
import time
import traceback
from datetime import date
from typing import Optional

from config import HISTORICAL_SEASONS, SPORTS, MAX_RETRIES, BACKOFF_BASE
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


# ── NBA ────────────────────────────────────────────────────────────────────────

def get_nba_game_logs(seasons: int = HISTORICAL_SEASONS) -> list:
    """
    Pull NBA game logs for the last `seasons` seasons.
    Returns list of dicts representing individual games.
    """
    try:
        from sportsreference.nba.schedule import Schedule
        from sportsreference.nba.teams import Teams
        import datetime

        records = []
        current_year = datetime.date.today().year
        season_years = range(current_year - seasons, current_year + 1)

        teams_obj = _retry_call(Teams)
        team_abbrevs = [t.abbreviation for t in teams_obj]

        for year in season_years:
            for abbrev in team_abbrevs:
                try:
                    schedule = _retry_call(Schedule, abbrev, year=year)
                    for game in schedule:
                        try:
                            records.append({
                                "sport": "NBA",
                                "season": year,
                                "team": abbrev,
                                "opponent": game.opponent_abbr,
                                "date": str(game.date),
                                "home": game.location == "Home",
                                "points": game.points,
                                "opp_points": game.opponent_points,
                                "win": game.result == "Win",
                                "fg_pct": game.field_goal_percentage,
                                "fg3_pct": game.three_point_field_goal_percentage,
                                "ft_pct": game.free_throw_percentage,
                                "rebounds": game.total_rebounds,
                                "assists": game.assists,
                                "turnovers": game.turnovers,
                                "steals": game.steals,
                                "blocks": game.blocks,
                            })
                        except Exception:
                            continue
                except Exception as exc:
                    log_error(
                        context=f"sports_ref.nba({abbrev}, {year})",
                        error_msg=str(exc),
                        tb=traceback.format_exc(),
                    )
                    continue
        return records
    except Exception as exc:
        log_error(
            context="sports_ref.get_nba_game_logs",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return []


# ── NFL ────────────────────────────────────────────────────────────────────────

def get_nfl_game_logs(seasons: int = HISTORICAL_SEASONS) -> list:
    """
    Pull NFL game logs for the last `seasons` seasons.
    """
    try:
        from sportsreference.nfl.schedule import Schedule
        from sportsreference.nfl.teams import Teams
        import datetime

        records = []
        current_year = datetime.date.today().year
        season_years = range(current_year - seasons, current_year + 1)

        teams_obj = _retry_call(Teams)
        team_abbrevs = [t.abbreviation for t in teams_obj]

        for year in season_years:
            for abbrev in team_abbrevs:
                try:
                    schedule = _retry_call(Schedule, abbrev, year=year)
                    for game in schedule:
                        try:
                            records.append({
                                "sport": "NFL",
                                "season": year,
                                "team": abbrev,
                                "opponent": game.opponent_abbr,
                                "date": str(game.date),
                                "home": game.location == "Home",
                                "points": game.points,
                                "opp_points": game.opponent_points,
                                "win": game.result == "Win",
                                "pass_yards": game.pass_yards,
                                "rush_yards": game.rush_yards,
                                "turnovers": game.turnovers,
                                "penalties": game.penalties,
                                "first_downs": game.first_downs,
                                "time_of_possession": game.time_of_possession,
                            })
                        except Exception:
                            continue
                except Exception as exc:
                    log_error(
                        context=f"sports_ref.nfl({abbrev}, {year})",
                        error_msg=str(exc),
                        tb=traceback.format_exc(),
                    )
                    continue
        return records
    except Exception as exc:
        log_error(
            context="sports_ref.get_nfl_game_logs",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return []


# ── MLB ────────────────────────────────────────────────────────────────────────

def get_mlb_game_logs(seasons: int = HISTORICAL_SEASONS) -> list:
    """
    Pull MLB game logs for the last `seasons` seasons.
    """
    try:
        from sportsreference.mlb.schedule import Schedule
        from sportsreference.mlb.teams import Teams
        import datetime

        records = []
        current_year = datetime.date.today().year
        season_years = range(current_year - seasons, current_year + 1)

        teams_obj = _retry_call(Teams)
        team_abbrevs = [t.abbreviation for t in teams_obj]

        for year in season_years:
            for abbrev in team_abbrevs:
                try:
                    schedule = _retry_call(Schedule, abbrev, year=year)
                    for game in schedule:
                        try:
                            records.append({
                                "sport": "MLB",
                                "season": year,
                                "team": abbrev,
                                "opponent": game.opponent_abbr,
                                "date": str(game.date),
                                "home": game.location == "Home",
                                "runs": game.runs,
                                "opp_runs": game.opponent_runs,
                                "win": game.result == "Win",
                                "hits": game.hits,
                                "errors": game.errors,
                                "batting_avg": game.batting_average,
                                "era": game.earned_run_average,
                                "strikeouts": game.strikeouts,
                                "walks": game.walks,
                            })
                        except Exception:
                            continue
                except Exception as exc:
                    log_error(
                        context=f"sports_ref.mlb({abbrev}, {year})",
                        error_msg=str(exc),
                        tb=traceback.format_exc(),
                    )
                    continue
        return records
    except Exception as exc:
        log_error(
            context="sports_ref.get_mlb_game_logs",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        return []


def get_all_historical_data() -> dict:
    """
    Pull all historical game logs for NBA, NFL, MLB.
    Returns dict: {sport: [game_dicts]}.
    """
    return {
        "NBA": get_nba_game_logs(),
        "NFL": get_nfl_game_logs(),
        "MLB": get_mlb_game_logs(),
    }
