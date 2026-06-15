"""
data/features.py — Feature engineering pipeline for NBA, NFL, MLB.
Combines ESPN live data, standings, recent form, H2H, and injury flags into
a flat feature vector suitable for XGBoost and Bayesian Network.
"""
import traceback
from datetime import date
from typing import Optional

import numpy as np

from data.scrapers.espn import get_scoreboard, get_standings, get_recent_results, get_team_stats
from data.scrapers.news import get_all_injury_flags
from db.supabase_client import log_error


# ── Feature definitions per sport ─────────────────────────────────────────────

NBA_FEATURES = [
    "home_wins", "home_losses", "home_win_pct", "home_points_for", "home_points_against",
    "away_wins", "away_losses", "away_win_pct", "away_points_for", "away_points_against",
    "home_streak", "away_streak",
    "home_recent_win_rate", "away_recent_win_rate",
    "home_avg_pts_scored", "home_avg_pts_allowed",
    "away_avg_pts_scored", "away_avg_pts_allowed",
    "home_injury_flag", "away_injury_flag",
    "home_field_advantage",
]

NFL_FEATURES = [
    "home_wins", "home_losses", "home_win_pct", "home_points_for", "home_points_against",
    "away_wins", "away_losses", "away_win_pct", "away_points_for", "away_points_against",
    "home_streak", "away_streak",
    "home_recent_win_rate", "away_recent_win_rate",
    "home_avg_pts_scored", "home_avg_pts_allowed",
    "away_avg_pts_scored", "away_avg_pts_allowed",
    "home_injury_flag", "away_injury_flag",
    "home_field_advantage",
]

MLB_FEATURES = [
    "home_wins", "home_losses", "home_win_pct", "home_points_for", "home_points_against",
    "away_wins", "away_losses", "away_win_pct", "away_points_for", "away_points_against",
    "home_streak", "away_streak",
    "home_recent_win_rate", "away_recent_win_rate",
    "home_avg_pts_scored", "home_avg_pts_allowed",
    "away_avg_pts_scored", "away_avg_pts_allowed",
    "home_injury_flag", "away_injury_flag",
    "home_field_advantage",
]

SPORT_FEATURES = {
    "NBA": NBA_FEATURES,
    "NFL": NFL_FEATURES,
    "MLB": MLB_FEATURES,
}


def _safe_mean(lst: list, key: str, default: float = 0.0) -> float:
    vals = [r.get(key, 0) for r in lst if r.get(key) is not None]
    return float(np.mean(vals)) if vals else default


def _recent_win_rate(results: list, n: int = 10) -> float:
    recent = results[-n:]
    if not recent:
        return 0.5
    return sum(1 for r in recent if r.get("win", False)) / len(recent)


def _streak(results: list) -> int:
    """Current win/loss streak (+N for win streak, -N for loss streak)."""
    if not results:
        return 0
    last_win = results[-1].get("win", False)
    streak = 0
    for r in reversed(results):
        if r.get("win", False) == last_win:
            streak += 1
        else:
            break
    return streak if last_win else -streak


def build_features_for_game(
    sport: str,
    home_team_id: str,
    away_team_id: str,
    home_team_name: str,
    away_team_name: str,
    standings: Optional[list] = None,
) -> dict:
    """
    Build the full feature dict for a game between home_team and away_team.
    Returns a flat dict keyed by feature name.
    """
    feature_keys = SPORT_FEATURES.get(sport, NBA_FEATURES)
    features = {k: 0.0 for k in feature_keys}

    try:
        # Standings
        if standings is None:
            standings = get_standings(sport)

        standing_map = {s["team_id"]: s for s in standings}
        home_s = standing_map.get(home_team_id, {})
        away_s = standing_map.get(away_team_id, {})

        features["home_wins"] = float(home_s.get("wins", 0))
        features["home_losses"] = float(home_s.get("losses", 0))
        features["home_win_pct"] = float(home_s.get("win_pct", 0.5))
        features["home_points_for"] = float(home_s.get("points_for", 0))
        features["home_points_against"] = float(home_s.get("points_against", 0))

        features["away_wins"] = float(away_s.get("wins", 0))
        features["away_losses"] = float(away_s.get("losses", 0))
        features["away_win_pct"] = float(away_s.get("win_pct", 0.5))
        features["away_points_for"] = float(away_s.get("points_for", 0))
        features["away_points_against"] = float(away_s.get("points_against", 0))

        # Recent form
        home_results = get_recent_results(sport, home_team_id, n=15)
        away_results = get_recent_results(sport, away_team_id, n=15)

        features["home_streak"] = float(_streak(home_results))
        features["away_streak"] = float(_streak(away_results))
        features["home_recent_win_rate"] = _recent_win_rate(home_results)
        features["away_recent_win_rate"] = _recent_win_rate(away_results)

        score_key = "points_scored" if sport in ("NBA", "NFL") else "points_scored"
        allow_key = "points_allowed"

        features["home_avg_pts_scored"] = _safe_mean(home_results, "points_scored")
        features["home_avg_pts_allowed"] = _safe_mean(home_results, "points_allowed")
        features["away_avg_pts_scored"] = _safe_mean(away_results, "points_scored")
        features["away_avg_pts_allowed"] = _safe_mean(away_results, "points_allowed")

        # Injury flags
        injury = get_all_injury_flags(sport, home_team_name, away_team_name)
        features["home_injury_flag"] = float(injury.get("home_injury_flag", 0))
        features["away_injury_flag"] = float(injury.get("away_injury_flag", 0))

        # Home field advantage constant
        features["home_field_advantage"] = 1.0

    except Exception as exc:
        log_error(
            context=f"features.build_features_for_game({sport}, {home_team_id} vs {away_team_id})",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )

    return features


def features_to_vector(features: dict, sport: str) -> np.ndarray:
    """Convert feature dict to ordered numpy array (for XGBoost)."""
    keys = SPORT_FEATURES.get(sport, NBA_FEATURES)
    return np.array([features.get(k, 0.0) for k in keys], dtype=np.float32)


def features_from_historical_record(record: dict, sport: str) -> dict:
    """
    Build a minimal feature dict from a sportsreference historical game record.
    Used for training data construction.
    """
    if sport == "NBA":
        return {
            "home_wins": 0.0,
            "home_losses": 0.0,
            "home_win_pct": 0.5,
            "home_points_for": float(record.get("points", 0)),
            "home_points_against": float(record.get("opp_points", 0)),
            "away_wins": 0.0,
            "away_losses": 0.0,
            "away_win_pct": 0.5,
            "away_points_for": float(record.get("opp_points", 0)),
            "away_points_against": float(record.get("points", 0)),
            "home_streak": 0.0,
            "away_streak": 0.0,
            "home_recent_win_rate": float(record.get("win", False)),
            "away_recent_win_rate": 1.0 - float(record.get("win", False)),
            "home_avg_pts_scored": float(record.get("points", 0)),
            "home_avg_pts_allowed": float(record.get("opp_points", 0)),
            "away_avg_pts_scored": float(record.get("opp_points", 0)),
            "away_avg_pts_allowed": float(record.get("points", 0)),
            "home_injury_flag": 0.0,
            "away_injury_flag": 0.0,
            "home_field_advantage": 1.0 if record.get("home", False) else 0.0,
        }
    elif sport == "NFL":
        return {
            "home_wins": 0.0,
            "home_losses": 0.0,
            "home_win_pct": 0.5,
            "home_points_for": float(record.get("points", 0)),
            "home_points_against": float(record.get("opp_points", 0)),
            "away_wins": 0.0,
            "away_losses": 0.0,
            "away_win_pct": 0.5,
            "away_points_for": float(record.get("opp_points", 0)),
            "away_points_against": float(record.get("points", 0)),
            "home_streak": 0.0,
            "away_streak": 0.0,
            "home_recent_win_rate": float(record.get("win", False)),
            "away_recent_win_rate": 1.0 - float(record.get("win", False)),
            "home_avg_pts_scored": float(record.get("points", 0)),
            "home_avg_pts_allowed": float(record.get("opp_points", 0)),
            "away_avg_pts_scored": float(record.get("opp_points", 0)),
            "away_avg_pts_allowed": float(record.get("points", 0)),
            "home_injury_flag": 0.0,
            "away_injury_flag": 0.0,
            "home_field_advantage": 1.0 if record.get("home", False) else 0.0,
        }
    elif sport == "MLB":
        return {
            "home_wins": 0.0,
            "home_losses": 0.0,
            "home_win_pct": 0.5,
            "home_points_for": float(record.get("runs", 0)),
            "home_points_against": float(record.get("opp_runs", 0)),
            "away_wins": 0.0,
            "away_losses": 0.0,
            "away_win_pct": 0.5,
            "away_points_for": float(record.get("opp_runs", 0)),
            "away_points_against": float(record.get("runs", 0)),
            "home_streak": 0.0,
            "away_streak": 0.0,
            "home_recent_win_rate": float(record.get("win", False)),
            "away_recent_win_rate": 1.0 - float(record.get("win", False)),
            "home_avg_pts_scored": float(record.get("runs", 0)),
            "home_avg_pts_allowed": float(record.get("opp_runs", 0)),
            "away_avg_pts_scored": float(record.get("opp_runs", 0)),
            "away_avg_pts_allowed": float(record.get("runs", 0)),
            "home_injury_flag": 0.0,
            "away_injury_flag": 0.0,
            "home_field_advantage": 1.0 if record.get("home", False) else 0.0,
        }
    return {}


def build_training_dataset(sport: str, game_logs: list) -> tuple:
    """
    Convert list of historical game log dicts into (X, y) arrays.
    X: np.ndarray of shape (n_samples, n_features)
    y: np.ndarray of shape (n_samples,) — 1 if home team won, 0 otherwise
    """
    X_rows = []
    y_rows = []

    for record in game_logs:
        feat = features_from_historical_record(record, sport)
        if not feat:
            continue
        vec = features_to_vector(feat, sport)
        label = 1 if record.get("win", False) else 0
        X_rows.append(vec)
        y_rows.append(label)

    if not X_rows:
        return np.empty((0, len(SPORT_FEATURES.get(sport, NBA_FEATURES)))), np.empty(0)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)
