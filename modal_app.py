"""
modal_app.py — Modal application entry point.
Defines all scheduled functions and the one-time initial_setup.
"""
import traceback
from datetime import datetime, date, timedelta

import modal

from config import (
    SPORTS, MODEL_VOLUME_NAME,
    SECRET_KALSHI, SECRET_SUPABASE, SECRET_SMTP,
    SECRET_API_SPORTS, SECRET_ODDS_API, SECRET_NEWS_API,
    MIN_ACCURACY_14D, ROLLING_ACCURACY_WINDOW,
)

# ── Modal image ────────────────────────────────────────────────────────────────

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
    )
    .add_local_python_source(
        "config",
        "data",
        "db",
        "models",
        "trading",
        "reporting",
    )
)

# ── Modal Volume for model persistence ────────────────────────────────────────
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

# ── Modal Secrets ─────────────────────────────────────────────────────────────
secrets = [
    modal.Secret.from_name(SECRET_KALSHI),
    modal.Secret.from_name(SECRET_SUPABASE),
    modal.Secret.from_name(SECRET_SMTP),
    modal.Secret.from_name(SECRET_API_SPORTS),
    modal.Secret.from_name(SECRET_ODDS_API),
    modal.Secret.from_name(SECRET_NEWS_API),
]

app = modal.App("kalshi-sports-bot")

VOLUME_PATH = "/models"


# ── Helper: full train for one sport ──────────────────────────────────────────

def _full_train_sport(sport: str):
    """Full training pipeline for a single sport."""
    from data.scrapers.sports_ref import get_nba_game_logs, get_nfl_game_logs, get_mlb_game_logs
    from models import xgboost_model, bayesian_model, meta_learner, calibration
    from data.features import build_training_dataset
    import numpy as np

    print(f"[{sport}] Pulling historical data...")
    if sport == "NBA":
        logs = get_nba_game_logs()
    elif sport == "NFL":
        logs = get_nfl_game_logs()
    elif sport == "MLB":
        logs = get_mlb_game_logs()
    else:
        return

    print(f"[{sport}] Got {len(logs)} historical game records")
    if not logs:
        print(f"[{sport}] No data — skipping")
        return

    print(f"[{sport}] Training XGBoost...")
    xgb_m = xgboost_model.train(sport, logs)
    xgboost_model.save_model(sport, xgb_m, VOLUME_PATH)

    print(f"[{sport}] Building Bayesian Network...")
    bn_m = bayesian_model.build_default_model()
    bn_m = bayesian_model.update_priors(bn_m, logs)
    bayesian_model.save_model(sport, bn_m, VOLUME_PATH)

    X, y = build_training_dataset(sport, logs)
    if X.shape[0] == 0:
        print(f"[{sport}] No features to train meta-learner")
        return

    print(f"[{sport}] Training meta-learner and calibrator...")
    xgb_probs = xgb_m.predict_proba(X)[:, 1]
    bn_probs = np.array([
        bayesian_model.predict_proba(bn_m, {"home_recent_win_rate": 0.5, "away_recent_win_rate": 0.5,
                                             "home_injury_flag": 0, "away_injury_flag": 0})
        for _ in range(X.shape[0])
    ])

    meta_m = meta_learner.train_meta_learner(xgb_probs, bn_probs, y)
    meta_learner.save_model(sport, meta_m, VOLUME_PATH)

    raw_probs = meta_m.predict_proba(np.column_stack([xgb_probs, bn_probs]))[:, 1]
    calib_m = calibration.train_calibrator(raw_probs, y)
    calibration.save_model(sport, calib_m, VOLUME_PATH)

    print(f"[{sport}] Full training complete — models saved to Volume")


# ── 1. Morning pipeline (9AM ET) ──────────────────────────────────────────────

@app.function(
    image=image,
    secrets=secrets,
    volumes={VOLUME_PATH: model_volume},
    schedule=modal.Cron("0 14 * * *"),
    timeout=1800,
)
def morning_pipeline():
    """9AM ET: scrape games, run parallel inference, place demo trades, log to Supabase."""
    run_date = datetime.utcnow().date()
    print(f"=== morning_pipeline {run_date} ===")

    from data.scrapers.espn import get_scoreboard
    from trading.executor import execute_for_sport
    from db.supabase_client import log_error

    all_summaries = []
    for sport in SPORTS:
        try:
            print(f"Scraping {sport} games...")
            games = get_scoreboard(sport, game_date=run_date)
            upcoming = [g for g in games if not g.get("completed", False)]
            print(f"{sport}: {len(upcoming)} upcoming games found")
            if not upcoming:
                continue
            summary = execute_for_sport(sport, upcoming, VOLUME_PATH, run_date)
            all_summaries.append(summary)
            print(f"{sport}: placed={summary['bets_placed']}, skipped={summary['bets_skipped']}")
        except Exception as exc:
            log_error(
                context=f"morning_pipeline({sport})",
                error_msg=str(exc),
                tb=traceback.format_exc(),
                run_date=run_date,
            )
            print(f"ERROR in {sport}: {exc}")
            continue

    print(f"morning_pipeline complete: {all_summaries}")
    model_volume.commit()


# ── 2. Nightly retrain (11PM ET) ──────────────────────────────────────────────

@app.function(
    image=image,
    secrets=secrets,
    volumes={VOLUME_PATH: model_volume},
    schedule=modal.Cron("0 4 * * *"),
    timeout=3600,
)
def nightly_retrain():
    """11PM ET: ingest resolved results, incremental retrain, check accuracy, save models."""
    run_date = datetime.utcnow().date()
    yesterday = run_date - timedelta(days=1)
    print(f"=== nightly_retrain {run_date} ===")

    from data.scrapers.espn import get_scoreboard
    from models import xgboost_model, bayesian_model
    from db.supabase_client import (
        log_error, log_model_performance, get_resolved_games_since
    )

    since_date = run_date - timedelta(days=ROLLING_ACCURACY_WINDOW)

    for sport in SPORTS:
        try:
            print(f"[{sport}] Fetching resolved results for {yesterday}...")
            completed = get_scoreboard(sport, game_date=yesterday)
            resolved = [g for g in completed if g.get("completed", False)]
            print(f"[{sport}] {len(resolved)} resolved games")

            new_logs = []
            for g in resolved:
                new_logs.append({
                    "sport": sport,
                    "home": True,
                    "win": g.get("home_winner", False),
                    "points": g.get("home_score", 0),
                    "opp_points": g.get("away_score", 0),
                    "home_recent_win_rate": 0.5,
                })

            xgb_m = xgboost_model.load_model(sport, VOLUME_PATH)
            bn_m = bayesian_model.load_model(sport, VOLUME_PATH)

            if xgb_m is None or bn_m is None:
                print(f"[{sport}] No existing models — skipping incremental retrain")
                continue

            if new_logs:
                print(f"[{sport}] Incremental XGBoost retrain on {len(new_logs)} new games...")
                xgb_m = xgboost_model.incremental_retrain(sport, new_logs, xgb_m)
                xgboost_model.save_model(sport, xgb_m, VOLUME_PATH)

                print(f"[{sport}] Updating BN priors...")
                bn_m = bayesian_model.update_priors(bn_m, new_logs)
                bayesian_model.save_model(sport, bn_m, VOLUME_PATH)

            history = get_resolved_games_since(sport, since_date)
            correct = 0
            total = len(history)
            for rec in history:
                predicted_win = (rec.get("final_prob") or 0.5) >= 0.5
                trade_status = rec.get("trade_status", "")
                actual_win = trade_status == "won" if trade_status in ("won", "lost") else predicted_win
                if predicted_win == actual_win:
                    correct += 1

            accuracy = correct / total if total > 0 else 0.5
            retrain_triggered = False

            print(f"[{sport}] 14-day accuracy: {accuracy:.3f} over {total} predictions")

            if accuracy < MIN_ACCURACY_14D and total >= 10:
                print(f"[{sport}] Accuracy {accuracy:.3f} below threshold — triggering full retrain")
                retrain_triggered = True
                _full_train_sport(sport)

            log_model_performance(
                run_date=run_date,
                sport=sport,
                accuracy_14d=accuracy,
                total_predictions=total,
                retrain_triggered=retrain_triggered,
            )

        except Exception as exc:
            log_error(
                context=f"nightly_retrain({sport})",
                error_msg=str(exc),
                tb=traceback.format_exc(),
                run_date=run_date,
            )
            print(f"ERROR in {sport}: {exc}")
            continue

    model_volume.commit()
    print("nightly_retrain complete")


# ── 3. Daily email (11:30PM ET) ───────────────────────────────────────────────

@app.function(
    image=image,
    secrets=secrets,
    schedule=modal.Cron("30 4 * * *"),
    timeout=300,
)
def send_daily_email():
    """11:30PM ET: build and send HTML daily summary email via SMTP."""
    run_date = datetime.utcnow().date()
    print(f"=== send_daily_email {run_date} ===")

    from reporting.email_report import send_daily_report
    from db.supabase_client import log_error

    try:
        send_daily_report(run_date)
        print("Email sent successfully")
    except Exception as exc:
        log_error(
            context="send_daily_email",
            error_msg=str(exc),
            tb=traceback.format_exc(),
            run_date=run_date,
        )
        print(f"Failed to send email: {exc}")


# ── 4. Initial setup (run once manually) ─────────────────────────────────────

@app.function(
    image=image,
    secrets=secrets,
    volumes={VOLUME_PATH: model_volume},
    timeout=7200,
)
def initial_setup():
    """
    One-time setup: verify Supabase tables reachable, pull 3-5 seasons of data,
    full-train models for NBA/NFL/MLB, save to Modal Volume.
    """
    print("=== initial_setup ===")

    from db.supabase_client import create_tables, log_error

    print("Verifying Supabase tables...")
    try:
        create_tables()
        print("Supabase OK")
    except Exception as exc:
        log_error(
            context="initial_setup.create_tables",
            error_msg=str(exc),
            tb=traceback.format_exc(),
        )
        print(f"Table check error: {exc}")

    print("Starting full model training for all sports...")
    for sport in SPORTS:
        try:
            _full_train_sport(sport)
        except Exception as exc:
            from db.supabase_client import log_error as _log_error
            _log_error(
                context=f"initial_setup._full_train_sport({sport})",
                error_msg=str(exc),
                tb=traceback.format_exc(),
            )
            print(f"ERROR training {sport}: {exc}")

    model_volume.commit()
    print("=== initial_setup complete ===")


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    print("Kalshi Sports Bot — use 'modal run modal_app.py::initial_setup' to initialize")
    print("Scheduled: morning_pipeline (9AM ET), nightly_retrain (11PM ET), send_daily_email (11:30PM ET)")
