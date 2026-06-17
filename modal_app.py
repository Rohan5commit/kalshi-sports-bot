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
        "basketball-reference-web-scraper>=0.8",
        "nfl_data_py>=0.3",
        "pybaseball>=2.2",
        "kaggle>=1.6",
        "lxml>=5.0",
        "feedparser>=6.0",
        "rapidfuzz>=3.0",
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
    from models import xgboost_model, bayesian_model, meta_learner, calibration
    from data.features import build_training_dataset, compute_elo_series
    from db.historical_store import upsert_elo_ratings
    import numpy as np

    print(f"[{sport}] {len(logs)} game logs — training")
    if len(logs) < 50:
        print(f"[{sport}] too few records — skipping")
        return

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
    print(f"[{sport}] Full training complete — models saved to Volume")


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_nba_data(incremental_from=None) -> list:
    from data.scrapers.nba import ingest_full_history, kaggle_to_game_logs, wyattowalsh_to_game_logs
    from data.scrapers.kaggle_loader import get_nba_games, get_nba_full, get_nba_shot_logs
    from db.historical_store import upsert_game_logs

    all_logs = []

    try:
        print("[NBA] Kaggle nathanlauga/nba-games...")
        df = get_nba_games()
        kaggle_logs = kaggle_to_game_logs(df)
        if incremental_from:
            kaggle_logs = [g for g in kaggle_logs if g.get("game_date", "") > str(incremental_from)]
        upsert_game_logs("NBA", kaggle_logs)
        all_logs.extend(kaggle_logs)
        print(f"[NBA] Kaggle nathanlauga: {len(kaggle_logs)} games")
    except Exception as exc:
        print(f"[NBA] Kaggle nathanlauga error: {exc}")

    try:
        print("[NBA] Kaggle wyattowalsh/basketball...")
        datasets = get_nba_full()
        game_df = datasets.get("game.csv", datasets.get(
            next((k for k in datasets if "game" in k.lower()), ""), None))
        if game_df is not None and not game_df.empty:
            wyatto_logs = wyattowalsh_to_game_logs(game_df)
            if incremental_from:
                wyatto_logs = [g for g in wyatto_logs if g.get("game_date", "") > str(incremental_from)]
            upsert_game_logs("NBA", wyatto_logs)
            all_logs.extend(wyatto_logs)
            print(f"[NBA] Kaggle wyattowalsh: {len(wyatto_logs)} games")
    except Exception as exc:
        print(f"[NBA] Kaggle wyattowalsh error: {exc}")

    try:
        print("[NBA] nba_data (basketball-reference.com)...")
        from data.scrapers.nba_data import ingest_full_history as nba_data_ingest
        from config import NBA_SEASONS
        nba_logs = nba_data_ingest(NBA_SEASONS, incremental_from=incremental_from)
        if nba_logs:
            upsert_game_logs("NBA", nba_logs)
            all_logs.extend(nba_logs)
            print(f"[NBA] nba_data: {len(nba_logs)} enriched games stored")
    except Exception as exc:
        print(f"[NBA] nba_data error: {exc}\n{traceback.format_exc()}")

    print(f"[NBA] Total unique game records: {len(all_logs)}")
    return all_logs


def _load_nfl_data(incremental_from=None) -> list:
    from data.scrapers.nfl import ingest_full_history
    from data.scrapers.kaggle_loader import get_nfl_scores, get_nfl_pbp, nfl_scores_to_game_logs
    from db.historical_store import upsert_game_logs

    all_logs = []

    try:
        print("[NFL] Kaggle tobycrabtree/nfl-scores...")
        scores_df = get_nfl_scores()
        kaggle_logs = nfl_scores_to_game_logs(scores_df)
        if incremental_from:
            kaggle_logs = [g for g in kaggle_logs if g.get("game_date", "") > str(incremental_from)]
        upsert_game_logs("NFL", kaggle_logs)
        all_logs.extend(kaggle_logs)
        print(f"[NFL] Kaggle NFL scores: {len(kaggle_logs)} games")
    except Exception as exc:
        print(f"[NFL] Kaggle NFL scores error: {exc}")

    print(f"[NFL] nfl_data_py: {NFL_SEASONS[0]}-{NFL_SEASONS[-1]} ({len(NFL_SEASONS)} seasons)")
    try:
        nfl_logs = ingest_full_history(
            seasons=NFL_SEASONS,
            incremental_from=incremental_from,
            pull_nextgen=False,
            pull_injuries=False,
            pull_contracts=False,
        )
        if nfl_logs:
            upsert_game_logs("NFL", nfl_logs)
            all_logs.extend(nfl_logs)
            print(f"[NFL] nfl_data_py: {len(nfl_logs)} enriched games stored")
    except Exception as exc:
        print(f"[NFL] nfl_data_py error: {exc}\n{traceback.format_exc()}")

    print(f"[NFL] Total unique game records: {len(all_logs)}")
    return all_logs


def _load_mlb_data(incremental_from=None) -> list:
    from data.scrapers.mlb import ingest_full_history, statcast_to_game_logs
    from data.scrapers.kaggle_loader import get_mlb_season_stats, get_mlb_game_logs, mlb_game_logs_to_standard
    from db.historical_store import upsert_game_logs

    all_logs = []

    try:
        print("[MLB] Kaggle saurabhshahane/mlb-game-log-dataset...")
        gl_df = get_mlb_game_logs()
        kaggle_logs = mlb_game_logs_to_standard(gl_df)
        if incremental_from:
            kaggle_logs = [g for g in kaggle_logs if g.get("game_date", "") > str(incremental_from)]
        upsert_game_logs("MLB", kaggle_logs)
        all_logs.extend(kaggle_logs)
        print(f"[MLB] Kaggle MLB logs: {len(kaggle_logs)} games")
    except Exception as exc:
        print(f"[MLB] Kaggle MLB logs error: {exc}")

    print(f"[MLB] pybaseball Statcast: {MLB_SEASONS[0]}-{MLB_SEASONS[-1]} ({len(MLB_SEASONS)} seasons)")
    try:
        mlb_logs = ingest_full_history(seasons=MLB_SEASONS, incremental_from=incremental_from)
        if mlb_logs:
            upsert_game_logs("MLB", mlb_logs)
            all_logs.extend(mlb_logs)
            print(f"[MLB] pybaseball: {len(mlb_logs)} enriched games stored")
    except Exception as exc:
        print(f"[MLB] pybaseball error: {exc}\n{traceback.format_exc()}")

    print(f"[MLB] Total unique game records: {len(all_logs)}")
    return all_logs


def _enrich_weather(sport: str, game_logs: list) -> list:
    from data.scrapers.weather import get_weather_for_game, compute_weather_impact_score
    from db.historical_store import get_cached_weather, upsert_weather_cache

    if sport == "NBA":
        return game_logs

    enriched_count = 0
    cache_writes = []
    for g in game_logs:
        home_team = g.get("home_team", "")
        game_date = g.get("game_date", "")
        if not home_team or not game_date:
            continue
        if g.get("weather_impact_score") is not None:
            continue
        venue_key = f"{sport}_{home_team}"
        cached = get_cached_weather(venue_key, game_date)
        if cached:
            w = cached
        else:
            w = get_weather_for_game(sport, home_team, game_date, is_future=False)
            if w:
                cache_writes.append({
                    "id": f"{venue_key}_{game_date}",
                    "venue": venue_key,
                    "game_date": game_date,
                    **{k: v for k, v in w.items() if v is not None},
                })
        if w:
            g["weather_impact_score"] = compute_weather_impact_score(w)
            enriched_count += 1

    if cache_writes:
        upsert_weather_cache(cache_writes)
    print(f"[{sport}] Weather enriched {enriched_count} games")
    return game_logs


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

    for sport in SPORTS:
        try:
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

            all_logs = get_game_logs(sport)
            xgb_m = xgboost_model.load_model(sport, VOLUME_PATH)
            bn_m = bayesian_model.load_model(sport, VOLUME_PATH)

            if xgb_m and bn_m and new_logs:
                xgb_m = xgboost_model.incremental_retrain(sport, new_logs, xgb_m)
                xgboost_model.save_model(sport, xgb_m, VOLUME_PATH)
                bn_m = bayesian_model.update_priors(bn_m, new_logs)
                bayesian_model.save_model(sport, bn_m, VOLUME_PATH)
                elos = compute_elo_series(all_logs)
                upsert_elo_ratings(sport, elos)

            since_date = run_date - timedelta(days=ROLLING_ACCURACY_WINDOW)
            history = get_resolved_games_since(sport, since_date)
            total = len(history)
            correct = sum(
                1 for r in history
                if ((r.get("final_prob") or 0.5) >= 0.5) ==
                   (r.get("trade_status") in ("won",))
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


# ── 4. Per-sport pipeline ─────────────────────────────────────────────────────

@app.function(
    image=image, secrets=secrets,
    volumes={VOLUME_PATH: model_volume},
    timeout=7200, cpu=8, memory=16384,
)
def _run_sport_pipeline(sport: str):
    if sport == "NBA":
        try:
            print("\n=== NBA DATA PIPELINE ===")
            logs = _load_nba_data()
            print(f"[NBA] Total logs: {len(logs)}")
            if logs:
                _train_models_for_sport("NBA", logs)
        except Exception as exc:
            print(f"NBA pipeline error: {exc}\n{traceback.format_exc()}")

    elif sport == "NFL":
        try:
            print("\n=== NFL DATA PIPELINE ===")
            logs = _load_nfl_data()
            print(f"[NFL] Total logs: {len(logs)}")
            if logs:
                _train_models_for_sport("NFL", logs)
        except Exception as exc:
            print(f"NFL pipeline error: {exc}\n{traceback.format_exc()}")

    elif sport == "MLB":
        try:
            print("\n=== MLB DATA PIPELINE ===")
            logs = _load_mlb_data()
            print(f"[MLB] Total logs: {len(logs)}")
            if logs:
                _train_models_for_sport("MLB", logs)
        except Exception as exc:
            print(f"MLB pipeline error: {exc}\n{traceback.format_exc()}")

    model_volume.commit()
    print(f"[{sport}] Pipeline complete")


# ── 5. Market data pipeline ───────────────────────────────────────────────────

@app.function(
    image=image, secrets=secrets,
    timeout=21600, cpu=8, memory=16384,
)
def _run_market_pipeline():
    """Ingest Kalshi + Polymarket historical data and match to game records."""
    from data.scrapers.kalshi_history import ingest_kalshi_history
    from data.scrapers.polymarket_history import ingest_polymarket_history

    print("\n=== KALSHI PIPELINE ===")
    try:
        k_stats = ingest_kalshi_history()
        print(f"[Kalshi] Complete: {k_stats}")
    except Exception as exc:
        print(f"[Kalshi] Pipeline error: {exc}\n{traceback.format_exc()}")

    print("\n=== POLYMARKET PIPELINE ===")
    try:
        p_stats = ingest_polymarket_history()
        print(f"[Polymarket] Complete: {p_stats}")
    except Exception as exc:
        print(f"[Polymarket] Pipeline error: {exc}\n{traceback.format_exc()}")

    print("[Market] Pipeline complete")


# ── 6. Sport retrain with market features ─────────────────────────────────────

@app.function(
    image=image, secrets=secrets,
    volumes={VOLUME_PATH: model_volume},
    timeout=3600, cpu=8, memory=16384,
)
def _run_sport_retrain(sport: str):
    """Retrain one sport model using existing game logs + newly ingested market features."""
    from db.historical_store import get_game_logs
    try:
        print(f"\n=== [{sport}] RETRAIN WITH MARKET FEATURES ===")
        logs = get_game_logs(sport)
        print(f"[{sport}] Loaded {len(logs)} logs from Supabase")
        if logs:
            _train_models_for_sport(sport, logs)
        else:
            print(f"[{sport}] No logs found — skipping")
    except Exception as exc:
        print(f"[{sport}] Retrain error: {exc}\n{traceback.format_exc()}")
    model_volume.commit()
    print(f"[{sport}] Market retrain complete")


# ── 7. Market retrain orchestrator ────────────────────────────────────────────

@app.function(
    image=image, secrets=secrets,
    volumes={VOLUME_PATH: model_volume},
    timeout=14400,
)
def run_market_retrain():
    """
    Targeted pipeline: ingest Kalshi+Polymarket data then retrain all models.
    Run this after initial_setup has already loaded game logs.
    Total time ~30-45 min (market ingestion ~15 min + parallel retrain ~15 min).
    """
    print("=== run_market_retrain ===")

    # Step 1: ingest market data (sequential — game matching needs game logs in Supabase)
    print("=== Step 1: Market data ingestion ===")
    _run_market_pipeline.remote()
    print("=== Market ingestion complete ===")

    # Step 2: retrain all 3 sports in parallel with market features enriched
    print("=== Step 2: Parallel retrain with market features ===")
    futures = [
        _run_sport_retrain.spawn("NBA"),
        _run_sport_retrain.spawn("NFL"),
        _run_sport_retrain.spawn("MLB"),
    ]
    for f in futures:
        f.get()

    model_volume.commit()
    print("\n=== run_market_retrain complete ===")


# ── 8. Initial setup ──────────────────────────────────────────────────────────

@app.function(
    image=image, secrets=secrets,
    volumes={VOLUME_PATH: model_volume},
    timeout=14400,  # 4h: sport pipelines (~45m) + market (~20m) + retrain (~15m)
)
def initial_setup():
    print("=== initial_setup ===")
    from db.supabase_client import create_tables

    try:
        create_tables()
        print("Supabase core tables OK")
    except Exception as exc:
        print(f"Table check: {exc}")

    # Phase 1: sport data ingestion + initial model training (parallel containers)
    print("=== Phase 1: NBA + NFL + MLB pipelines (parallel) ===")
    futures = [
        _run_sport_pipeline.spawn("NBA"),
        _run_sport_pipeline.spawn("NFL"),
        _run_sport_pipeline.spawn("MLB"),
    ]
    for f in futures:
        f.get()
    print("=== Phase 1 complete ===")

    # Phase 2: market data ingestion (after game logs exist in Supabase for matching)
    print("=== Phase 2: Market data ingestion ===")
    try:
        _run_market_pipeline.remote()
        print("=== Phase 2 complete ===")
    except Exception as exc:
        print(f"Phase 2 market ingestion error: {exc}")

    # Phase 3: retrain all sports with Kalshi+Polymarket features
    print("=== Phase 3: Retrain with market features (parallel) ===")
    futures = [
        _run_sport_retrain.spawn("NBA"),
        _run_sport_retrain.spawn("NFL"),
        _run_sport_retrain.spawn("MLB"),
    ]
    for f in futures:
        f.get()
    print("=== Phase 3 complete ===")

    print("\n=== initial_setup complete ===")


@app.local_entrypoint()
def main():
    print("Kalshi Sports Bot — run initial_setup once, then scheduled jobs take over")

