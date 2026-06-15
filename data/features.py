"""
data/features.py — Enhanced feature engineering pipeline.
Features per sport: rolling 5/10/20-game win rates, Elo diff, rest day diff,
back-to-back flags, referee/umpire tendency, pitcher matchup (MLB),
lineup confirmation (Rotowire+ESPN), injury flags (Currents+Rotowire).
"""
import traceback
from typing import Optional

import numpy as np

from db.supabase_client import log_error

# ── Feature definitions ────────────────────────────────────────────────────────

NBA_FEATURES = [
    "home_win_pct", "away_win_pct",
    "home_5g_win_rate", "away_5g_win_rate",
    "home_10g_win_rate", "away_10g_win_rate",
    "home_20g_win_rate", "away_20g_win_rate",
    "home_elo", "away_elo", "elo_diff",
    "home_rest_days", "away_rest_days", "rest_day_diff",
    "home_back_to_back", "away_back_to_back",
    "home_avg_score_10g", "away_avg_score_10g",
    "home_avg_allowed_10g", "away_avg_allowed_10g",
    "home_usg_pct", "away_usg_pct",
    "home_ts_pct", "away_ts_pct",
    "home_home_win_rate", "away_away_win_rate",
    "home_injury_flag", "away_injury_flag",
    "home_lineup_confirmed", "away_lineup_confirmed",
    "home_field_advantage",
]

NFL_FEATURES = [
    "home_win_pct", "away_win_pct",
    "home_5g_win_rate", "away_5g_win_rate",
    "home_10g_win_rate", "away_10g_win_rate",
    "home_elo", "away_elo", "elo_diff",
    "home_rest_days", "away_rest_days", "rest_day_diff",
    "home_off_epa", "away_off_epa",
    "home_def_epa", "away_def_epa",
    "home_avg_score_5g", "away_avg_score_5g",
    "home_avg_allowed_5g", "away_avg_allowed_5g",
    "weather_temp", "weather_wind",
    "surface_grass", "roof_outdoor",
    "home_injury_flag", "away_injury_flag",
    "home_field_advantage",
]

MLB_FEATURES = [
    "home_win_pct", "away_win_pct",
    "home_5g_win_rate", "away_5g_win_rate",
    "home_10g_win_rate", "away_10g_win_rate",
    "home_elo", "away_elo", "elo_diff",
    "home_rest_days", "away_rest_days",
    "home_exit_velocity", "away_exit_velocity",
    "home_hard_hit_rate", "away_hard_hit_rate",
    "home_pitcher_whiff_rate", "away_pitcher_whiff_rate",
    "park_factor", "umpire_run_factor",
    "pitcher_matchup_win_rate",
    "home_avg_score_10g", "away_avg_score_10g",
    "home_injury_flag", "away_injury_flag",
    "home_lineup_confirmed", "away_lineup_confirmed",
    "home_field_advantage",
]

SPORT_FEATURES = {"NBA": NBA_FEATURES, "NFL": NFL_FEATURES, "MLB": MLB_FEATURES}

DEFAULT_ELO = 1500.0


# ── Elo utilities ──────────────────────────────────────────────────────────────

def update_elo(home_elo: float, away_elo: float, home_win: bool, k: float = 20.0):
    exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))
    delta = k * ((1.0 if home_win else 0.0) - exp_home)
    return home_elo + delta, away_elo - delta


def compute_elo_series(game_logs: list) -> dict:
    """
    Replay all games chronologically to compute final Elo ratings.
    game_logs must be sorted by game_date ascending.
    Returns {team: elo}.
    """
    elos: dict[str, float] = {}
    for g in game_logs:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        if not home or not away:
            continue
        h_elo = elos.get(home, DEFAULT_ELO)
        a_elo = elos.get(away, DEFAULT_ELO)
        h_elo, a_elo = update_elo(h_elo, a_elo, bool(g.get("home_win", False)))
        elos[home] = h_elo
        elos[away] = a_elo
    return elos


# ── Rolling helpers ────────────────────────────────────────────────────────────

def _win_rate(history: list, n: int) -> float:
    recent = history[-n:] if len(history) >= n else history
    if not recent:
        return 0.5
    return sum(1 for g in recent if g.get("win", False)) / len(recent)


def _avg(history: list, key: str, n: int, default: float = 0.0) -> float:
    recent = history[-n:] if len(history) >= n else history
    vals = [g[key] for g in recent if key in g and g[key] is not None]
    return float(np.mean(vals)) if vals else default


# ── Training dataset (reads from Supabase historical tables) ──────────────────

def build_training_dataset(sport: str, game_logs: list = None) -> tuple:
    """
    Build (X, y) arrays from historical game logs.
    game_logs: pre-loaded list (used when called from initial_setup);
               if None, reads from Supabase historical_store.
    Computes rolling features with strict no-lookahead ordering.
    """
    if game_logs is None:
        from db.historical_store import get_game_logs
        game_logs = get_game_logs(sport)

    game_logs = sorted(game_logs, key=lambda g: g.get("game_date", ""))

    team_history: dict[str, list] = {}
    team_elo: dict[str, float] = {}
    team_home_wins: dict[str, list] = {}
    team_away_wins: dict[str, list] = {}

    X_rows, y_rows = [], []
    feature_keys = SPORT_FEATURES[sport]

    for g in game_logs:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        if not home or not away:
            continue

        h_hist = team_history.get(home, [])
        a_hist = team_history.get(away, [])
        h_elo = team_elo.get(home, DEFAULT_ELO)
        a_elo = team_elo.get(away, DEFAULT_ELO)
        home_win = bool(g.get("home_win", g.get("win", False)))

        feat = _build_feat(sport, g, h_hist, a_hist, h_elo, a_elo,
                           team_home_wins.get(home, []),
                           team_away_wins.get(away, []))

        X_rows.append([feat.get(k, 0.0) for k in feature_keys])
        y_rows.append(1 if home_win else 0)

        # Update rolling state
        h_elo, a_elo = update_elo(h_elo, a_elo, home_win)
        team_elo[home] = h_elo
        team_elo[away] = a_elo

        for team, is_home, win, score, opp in [
            (home, True, home_win, g.get("home_score", 0), g.get("away_score", 0)),
            (away, False, not home_win, g.get("away_score", 0), g.get("home_score", 0)),
        ]:
            if team not in team_history:
                team_history[team] = []
                team_home_wins[team] = []
                team_away_wins[team] = []
            team_history[team].append({
                "win": win, "score": score, "opp_score": opp,
                "rest_days": g.get(f"{'home' if is_home else 'away'}_rest_days", 7),
                "back_to_back": g.get(f"{'home' if is_home else 'away'}_back_to_back", False),
            })
            if is_home:
                team_home_wins[team].append(win)
            else:
                team_away_wins[team].append(win)

    n = len(X_rows)
    if n == 0:
        return np.empty((0, len(feature_keys)), dtype=np.float32), np.empty(0, dtype=np.int32)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


def _build_feat(sport: str, g: dict, h_hist: list, a_hist: list,
                h_elo: float, a_elo: float,
                h_home_wins: list, a_away_wins: list) -> dict:
    """Build feature dict from pre-game history (no lookahead)."""
    f: dict = {}

    # Win rates — rolling 5/10/20
    all_h = [r["win"] for r in h_hist]
    all_a = [r["win"] for r in a_hist]
    n_h, n_a = len(all_h), len(all_a)

    f["home_win_pct"] = sum(all_h) / n_h if n_h else 0.5
    f["away_win_pct"] = sum(all_a) / n_a if n_a else 0.5
    f["home_5g_win_rate"] = _win_rate(h_hist, 5)
    f["away_5g_win_rate"] = _win_rate(a_hist, 5)
    f["home_10g_win_rate"] = _win_rate(h_hist, 10)
    f["away_10g_win_rate"] = _win_rate(a_hist, 10)
    f["home_20g_win_rate"] = _win_rate(h_hist, 20)
    f["away_20g_win_rate"] = _win_rate(a_hist, 20)

    # Elo
    f["home_elo"] = h_elo / 2000.0  # normalise to ~[0,1]
    f["away_elo"] = a_elo / 2000.0
    f["elo_diff"] = (h_elo - a_elo) / 400.0

    # Rest / schedule
    h_rest = g.get("home_rest_days", 7)
    a_rest = g.get("away_rest_days", 7)
    f["home_rest_days"] = min(h_rest, 14) / 14.0
    f["away_rest_days"] = min(a_rest, 14) / 14.0
    f["rest_day_diff"] = (h_rest - a_rest) / 14.0
    f["home_back_to_back"] = float(bool(g.get("home_back_to_back", False)))
    f["away_back_to_back"] = float(bool(g.get("away_back_to_back", False)))

    # Scoring averages
    f["home_avg_score_10g"] = _avg(h_hist, "score", 10) / 150.0
    f["away_avg_score_10g"] = _avg(a_hist, "score", 10) / 150.0
    f["home_avg_allowed_10g"] = _avg(h_hist, "opp_score", 10) / 150.0
    f["away_avg_allowed_10g"] = _avg(a_hist, "opp_score", 10) / 150.0
    f["home_avg_score_5g"] = _avg(h_hist, "score", 5) / 50.0
    f["away_avg_score_5g"] = _avg(a_hist, "score", 5) / 50.0
    f["home_avg_allowed_5g"] = _avg(h_hist, "opp_score", 5) / 50.0
    f["away_avg_allowed_5g"] = _avg(a_hist, "opp_score", 5) / 50.0

    # Home/away splits
    f["home_home_win_rate"] = (sum(h_home_wins) / len(h_home_wins)) if h_home_wins else 0.5
    f["away_away_win_rate"] = (sum(a_away_wins) / len(a_away_wins)) if a_away_wins else 0.5

    # Sport-specific stored metrics
    if sport == "NBA":
        f["home_usg_pct"] = float(g.get("home_usg_pct") or 0.20)
        f["away_usg_pct"] = float(g.get("away_usg_pct") or 0.20)
        f["home_ts_pct"] = float(g.get("home_ts_pct") or 0.55)
        f["away_ts_pct"] = float(g.get("away_ts_pct") or 0.55)

    elif sport == "NFL":
        f["home_off_epa"] = float(g.get("home_off_epa") or 0.0)
        f["away_off_epa"] = float(g.get("away_off_epa") or 0.0)
        f["home_def_epa"] = float(g.get("home_def_epa") or 0.0)
        f["away_def_epa"] = float(g.get("away_def_epa") or 0.0)
        temp = g.get("weather_temp")
        wind = g.get("weather_wind")
        f["weather_temp"] = (float(temp) / 100.0) if temp is not None else 0.7
        f["weather_wind"] = (float(wind) / 30.0) if wind is not None else 0.1
        f["surface_grass"] = 1.0 if "grass" in str(g.get("surface", "")).lower() else 0.0
        f["roof_outdoor"] = 1.0 if "out" in str(g.get("roof", "")).lower() else 0.0

    elif sport == "MLB":
        f["home_exit_velocity"] = (float(g.get("home_exit_velocity") or 88.0) - 80) / 20.0
        f["away_exit_velocity"] = (float(g.get("away_exit_velocity") or 88.0) - 80) / 20.0
        f["home_hard_hit_rate"] = float(g.get("home_hard_hit_rate") or 0.35)
        f["away_hard_hit_rate"] = float(g.get("away_hard_hit_rate") or 0.35)
        f["home_pitcher_whiff_rate"] = float(g.get("home_pitcher_whiff_rate") or 0.25)
        f["away_pitcher_whiff_rate"] = float(g.get("away_pitcher_whiff_rate") or 0.25)
        f["park_factor"] = float(g.get("park_factor") or 1.0)
        f["umpire_run_factor"] = float(g.get("umpire_run_factor") or 1.0)
        f["pitcher_matchup_win_rate"] = float(g.get("pitcher_matchup_win_rate") or 0.5)

    # Injury / lineup (default 0 for training data; populated at inference time)
    f["home_injury_flag"] = float(g.get("home_injury_flag") or 0)
    f["away_injury_flag"] = float(g.get("away_injury_flag") or 0)
    f["home_lineup_confirmed"] = float(g.get("home_lineup_confirmed") or 0)
    f["away_lineup_confirmed"] = float(g.get("away_lineup_confirmed") or 0)
    f["home_field_advantage"] = 1.0

    return f


# ── Live inference feature builder ────────────────────────────────────────────

def build_features_for_game(
    sport: str,
    home_team: str,
    away_team: str,
    game_meta: Optional[dict] = None,
) -> dict:
    """
    Build features for a live upcoming game.
    Reads recent game logs from Supabase, Elo from elo_ratings table,
    injury/lineup from Rotowire + Currents.
    """
    from db.historical_store import get_game_logs, get_elo_ratings
    from data.scrapers.rotowire import get_injury_flags, get_lineup_confirmation
    from data.scrapers.news import get_all_injury_flags

    feature_keys = SPORT_FEATURES[sport]
    game_meta = game_meta or {}

    try:
        logs = get_game_logs(sport, limit=2000)
        elos = get_elo_ratings(sport)

        team_history: dict[str, list] = {}
        team_elo: dict[str, float] = {}
        team_home_wins: dict[str, list] = {}
        team_away_wins: dict[str, list] = {}

        for g in sorted(logs, key=lambda x: x.get("game_date", "")):
            ht, at = g.get("home_team", ""), g.get("away_team", "")
            hw = bool(g.get("home_win", False))
            for team, is_home, win, score, opp in [
                (ht, True, hw, g.get("home_score", 0), g.get("away_score", 0)),
                (at, False, not hw, g.get("away_score", 0), g.get("home_score", 0)),
            ]:
                if team not in team_history:
                    team_history[team] = []
                    team_home_wins[team] = []
                    team_away_wins[team] = []
                team_history[team].append({
                    "win": win, "score": score, "opp_score": opp,
                    "rest_days": g.get(f"{'home' if is_home else 'away'}_rest_days", 7),
                    "back_to_back": g.get(f"{'home' if is_home else 'away'}_back_to_back", False),
                })
                (team_home_wins if is_home else team_away_wins)[team].append(win)

        h_elo = elos.get(home_team, DEFAULT_ELO)
        a_elo = elos.get(away_team, DEFAULT_ELO)

        feat = _build_feat(
            sport, game_meta,
            team_history.get(home_team, []),
            team_history.get(away_team, []),
            h_elo, a_elo,
            team_home_wins.get(home_team, []),
            team_away_wins.get(away_team, []),
        )

        # Live injury/lineup signals
        try:
            rw_inj = get_injury_flags(sport, home_team, away_team)
            news_inj = get_all_injury_flags(sport, home_team, away_team)
            feat["home_injury_flag"] = float(max(rw_inj.get("home_injury_flag", 0),
                                                  news_inj.get("home_injury_flag", 0)))
            feat["away_injury_flag"] = float(max(rw_inj.get("away_injury_flag", 0),
                                                  news_inj.get("away_injury_flag", 0)))
            lineup = get_lineup_confirmation(sport, home_team, away_team)
            feat["home_lineup_confirmed"] = float(lineup.get("home_lineup_confirmed", 0))
            feat["away_lineup_confirmed"] = float(lineup.get("away_lineup_confirmed", 0))
        except Exception:
            pass

        return feat

    except Exception as exc:
        log_error(
            context=f"features.build_features_for_game({sport}, {home_team} vs {away_team})",
            error_msg=str(exc), tb=traceback.format_exc(),
        )
        return {k: 0.0 for k in feature_keys}


def features_to_vector(features: dict, sport: str) -> np.ndarray:
    keys = SPORT_FEATURES.get(sport, NBA_FEATURES)
    return np.array([features.get(k, 0.0) for k in keys], dtype=np.float32)
