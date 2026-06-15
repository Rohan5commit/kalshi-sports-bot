"""
modal_app.py — Modal application entry point.
"""
import traceback
from datetime import datetime, timedelta

import modal

from config import (
    SPORTS, MODEL_VOLUME_NAME,
    SECRET_KALSHI, SECRET_SUPABASE, SECRET_SMTP,
    SECRET_API_SPORTS, SECRET_NEWS_API, SECRET_KAGGLE,
    MIN_ACCURACY_14D, ROLLING_ACCURACY_WINDOW,
    NBA_SEASONS, NFL_SEASONS, MLB_SEASONS,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "xgboost>=2.0",
        "pgmpy>=0.1.24",
        "scikit-learn>=1.4",
        "numpy>=1.26.0",
        "pandas>=2.0",
        "requests>=2.31",
        "supabase>=2.0",
        "sportsreference>=0.5.0",
        "scipy>=1.12",
        "nba_api>=1.4",
        "nfl_data_py>=0.3",
        "pybaseball>=2.2",
        "kaggle>=1.6",
    )
    .add_local_python_source(
        "config", "data", "db", "models", "trading", "reporting",
    )
)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

secrets = [
    modal.Secret.from_name(SECRET_KALSHI),
    modal.Secret.from_name(SECRET_SUPABASE),
    modal.Secret.from_name(SECRET_SMTP),
    modal.Secret.from_name(SECRET_API_SPORTS),
    modal.Secret.from_name(SECRET_NEWS_API),
    modal.Secret.from_name(SECRET_KAGGLE),
]

app = modal.App("kalshi-sports-bot")
VOLUME_PATH = "/models"


# ── Training helper ────────────────────────────────────────────────────────────

def _train_models_for_sport(sport: str, logs: list):
    """Full train pipeline for one sport given pre-loaded game logs."""
    from models import xgboost_model, bayesian_model, meta_learner, calibration
    from data.features import build_training_dataset, compute_elo_series
    from db.historical_store import upsert_elo_ratings
    import numpy as np

    print(f"[{sport}] {len(logs)} game logs — training")
    if len(logs) < 50:
        print(f"[{sport}] too few records — skipping")
        return

    # Compute and persist Elo ratings
    elos = compute_elo_series(logs)
    upsert_elo_ratings(sport, elos)

    X, y = build_training_dataset(sport, logs)
    if X.shape[0] == 0:
        print(f"[{sport}] no feature rows — skipping")
        return

    print(f"[{sport}] Training XGBoost on {X.shape[0]} samples, {X.shape[1]} features")
    xgb_m = xgboost_model.train(sport, logs)
    xgboost_model.save_model(sport, xgb_m, VOLUME_PATH)

    print(f"[{sport}] Building Bayesian Network")
    bn_m = bayesian_model.build_default_model()
    bn_m = bayesian_model.update_priors(bn_m, logs)
    bayesian_model.save_model(sport, bn_m, VOLUME_PATH)

    xgb_probs = xgb_m.predict_proba(X)[:, 1]
    bn_probs = np.array([
        bayesian_model.predict_proba(bn_m, {
            "home_recent_win_rate": 0.5, "away_recent_win_rate": 0.5,
            "home_injury_flag": 0, "away_injury_flag": 0,
        }) for _ in range(X.shape[0])
    ])

    meta_m = meta_learner.train_meta_learner(xgb_probs, bn_probs, y)
    meta_learner.save_model(sport, meta_m, VOLUME_PATH)

    raw_probs = meta_m.predict_proba(np.column_stack([xgb_probs, bn_probs]))[:, 1]
    calib_m = calibration.train_calibrator(raw_probs, y)
    calibration.save_model(sport, calib_m, VOLUME_PATH)
    print(f"[{sport}] Models saved to Volume")


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_nba_data(incremental_from=None) -> list:
    from data.scrapers.nba import get_season_games, add_rest_and_b2b, kaggle_to_game_logs
    from data.scrapers.kaggle_loader import get_nba_games
    from db.historical_store import upsert_game_logs

    all_logs = []

    # Kaggle bulk load
    try:
        print("[NBA] Downloading Kaggle dataset...")
        df = get_nba_games()
        kaggle_logs = kaggle_to_game_logs(df)
        if incremental_from:
            kaggle_logs = [g for g in kaggle_logs if g.get("game_date", "") > str(incremental_from)]
        print(f"[NBA] Kaggle: {len(kaggle_logs)} games")
        upsert_game_logs("NBA", kaggle_logs)
        all_logs.extend(kaggle_logs)
    except Exception as exc:
        print(f"[NBA] Kaggle error: {exc}")

    # nba_api for recent seasons (more detail: rest days, officials)
    for season in NBA_SEASONS:
        try:
            print(f"[NBA] nba_api season {season}...")
            games = get_season_games(season)
            games = add_rest_and_b2b(games)
            if incremental_from:
                games = [g for g in games if g.get("game_date", "") > str(incremental_from)]
            print(f"[NBA] nba_api {season}: {len(games)} games")
            upsert_game_logs("NBA", games)
            all_logs.extend(games)
        except Exception as exc:
            print(f"[NBA] nba_api season {season} error: {exc}")

    return all_logs


def _load_nfl_data(incremental_from=None) -> list:
    from data.scrapers.nfl import get_schedules, schedules_to_game_logs, get_pbp, compute_epa_metrics
    from data.scrapers.kaggle_loader import get_nfl_pbp
    from db.historical_store import upsert_game_logs

    all_logs = []

    # nfl_data_py schedules (primary)
    try:
        print("[NFL] nfl_data_py schedules...")
        df = get_schedules(NFL_SEASONS)
        logs = schedules_to_game_logs(df)
        if incremental_from:
            logs = [g for g in logs if g.get("game_date", "") > str(incremental_from)]
        print(f"[NFL] nfl_data_py: {len(logs)} games")
        upsert_game_logs("NFL", logs)
        all_logs.extend(logs)
    except Exception as exc:
        print(f"[NFL] nfl_data_py error: {exc}")

    return all_logs


def _load_mlb_data(incremental_from=None) -> list:
    from data.scrapers.mlb import get_statcast_season_chunked, statcast_to_game_logs
    from db.historical_store import upsert_game_logs

    all_logs = []
    for year in MLB_SEASONS:
        try:
            print(f"[MLB] Statcast {year}...")
            df = get_statcast_season_chunked(year)
            logs = statcast_to_game_logs(df)
            if incremental_from:
                logs = [g for g in logs if g.get("game_date", "") > str(incremental_from)]
            print(f"[MLB] Statcast {year}: {len(logs)} games")
            upsert_game_logs("MLB", logs)
            all_logs.extend(logs)
        except Exception as exc:
            print(f"[MLB] Statcast {year} error: {exc}")

    return all_logs


# ── 1. Morning pipeline ────────────────────────────────────────────────────────

@app.function(
    image=image, secrets=secrets,
    volumes={VOLUME_PATH: model_volume},
    schedule=modal.Cron("0 14 * * *"), timeout=1800,
)
def morning_pipeline():
    run_date = datetime.utcnow().date()
    print(f"=== morning_pipeline {run_date} ===")

    from data.scrapers.espn import get_scoreboard
    from trading.executor import execute_for_sport
    from db.supabase_client import log_error

    for sport in SPORTS:
        try:
            games = get_scoreboard(sport, game_date=run_date)
            upcoming = [g for g in games if not g.get("completed", False)]
            print(f"{sport}: {len(upcoming)} upcoming games")
            if not upcoming:
                continue
            summary = execute_for_sport(sport, upcoming, VOLUME_PATH, run_date)
            print(f"{sport}: placed={summary['bets_placed']}, skipped={summary['bets_skipped']}")
        except Exception as exc:
            log_error(context=f"morning_pipeline({sport})", error_msg=str(exc),
                      tb=traceback.format_exc(), run_date=run_date)
            print(f"ERROR {sport}: {exc}")

    model_volume.commit()
    print("morning_pipeline complete")


# ── 2. Nightly retrain ────────────────────────────────────────────────────────

@app.function(
    image=image, secrets=secrets,
    volumes={VOLUME_PATH: model_volume},
    schedule=modal.Cron("0 4 * * *"), timeout=3600,
)
def nightly_retrain():
    run_date = datetime.utcnow().date()
    yesterday = run_date - timedelta(days=1)
    print(f"=== nightly_retrain {run_date} ===")

    from data.scrapers.espn import get_scoreboard
    from models import xgboost_model, bayesian_model
    from db.supabase_client import log_error, log_model_performance, get_resolved_games_since
    from db.historical_store import get_last_game_date, upsert_game_logs, get_game_logs, upsert_elo_ratings
    from data.features import compute_elo_series

    since_date = run_date - timedelta(days=ROLLING_ACCURACY_WINDOW)

    for sport in SPORTS:
        try:
            # Fetch new ESPN results and upsert
            completed = get_scoreboard(sport, game_date=yesterday)
            new_logs = [
                {
                    "game_id": f"espn_{sport}_{g.get('game_id', '')}_{yesterday}",
                    "game_date": str(yesterday),
                    "season": yesterday.year,
                    "home_team": g.get("home_team", ""),
                    "away_team": g.get("away_team", ""),
                    "home_score": int(g.get("home_score", 0)),
                    "away_score": int(g.get("away_score", 0)),
                    "home_win": bool(g.get("home_winner", False)),
                    "source": "espn",
                }
                for g in completed if g.get("completed", False)
            ]
            if new_logs:
                upsert_game_logs(sport, new_logs)
                print(f"[{sport}] upserted {len(new_logs)} new results")

            # Incremental retrain
            all_logs = get_game_logs(sport)
            xgb_m = xgboost_model.load_model(sport, VOLUME_PATH)
            bn_m = bayesian_model.load_model(sport, VOLUME_PATH)

            if xgb_m and bn_m and new_logs:
                xgb_m = xgboost_model.incremental_retrain(sport, new_logs, xgb_m)
                xgboost_model.save_model(sport, xgb_m, VOLUME_PATH)
                bn_m = bayesian_model.update_priors(bn_m, new_logs)
                bayesian_model.save_model(sport, bn_m, VOLUME_PATH)
                # Refresh Elo
                elos = compute_elo_series(all_logs)
                upsert_elo_ratings(sport, elos)

            # Accuracy check
            history = get_resolved_games_since(sport, since_date)
            total = len(history)
            correct = sum(
                1 for r in history
                if ((r.get("final_prob") or 0.5) >= 0.5) ==
                   (r.get("trade_status") == "won" if r.get("trade_status") in ("won", "lost")
                    else (r.get("final_prob") or 0.5) >= 0.5)
            )
            accuracy = correct / total if total > 0 else 0.5
            retrain_triggered = False
            print(f"[{sport}] 14d accuracy: {accuracy:.3f} ({total} preds)")

            if accuracy < MIN_ACCURACY_14D and total >= 10:
                print(f"[{sport}] Full retrain triggered")
                retrain_triggered = True
                _train_models_for_sport(sport, all_logs)

            log_model_performance(run_date=run_date, sport=sport,
                                  accuracy_14d=accuracy, total_predictions=total,
                                  retrain_triggered=retrain_triggered)

        except Exception as exc:
            log_error(context=f"nightly_retrain({sport})", error_msg=str(exc),
                      tb=traceback.format_exc(), run_date=run_date)
            print(f"ERROR {sport}: {exc}")

    model_volume.commit()
    print("nightly_retrain complete")


# ── 3. Daily email ────────────────────────────────────────────────────────────

@app.function(
    image=image, secrets=secrets,
    schedule=modal.Cron("30 4 * * *"), timeout=300,
)
def send_daily_email():
    run_date = datetime.utcnow().date()
    print(f"=== send_daily_email {run_date} ===")
    from reporting.email_report import send_daily_report
    from db.supabase_client import log_error
    try:
        send_daily_report(run_date)
        print("Email sent")
    except Exception as exc:
        log_error(context="send_daily_email", error_msg=str(exc),
                  tb=traceback.format_exc(), run_date=run_date)


# ── 4. Initial setup ──────────────────────────────────────────────────────────

@app.function(
    image=image, secrets=secrets,
    volumes={VOLUME_PATH: model_volume},
    timeout=14400,  # 4 hours for large Statcast downloads
)
def initial_setup():
    print("=== initial_setup ===")
    from db.supabase_client import create_tables

    try:
        create_tables()
        print("Supabase OK")
    except Exception as exc:
        print(f"Table check: {exc}")

    # NBA
    try:
        nba_logs = _load_nba_data()
        _train_models_for_sport("NBA", nba_logs)
    except Exception as exc:
        print(f"NBA pipeline error: {exc}\n{traceback.format_exc()}")

    # NFL
    try:
        nfl_logs = _load_nfl_data()
        _train_models_for_sport("NFL", nfl_logs)
    except Exception as exc:
        print(f"NFL pipeline error: {exc}\n{traceback.format_exc()}")

    # MLB
    try:
        mlb_logs = _load_mlb_data()
        _train_models_for_sport("MLB", mlb_logs)
    except Exception as exc:
        print(f"MLB pipeline error: {exc}\n{traceback.format_exc()}")

    model_volume.commit()
    print("=== initial_setup complete ===")


@app.local_entrypoint()
def main():
    print("Kalshi Sports Bot — run initial_setup once, then scheduled jobs take over")
