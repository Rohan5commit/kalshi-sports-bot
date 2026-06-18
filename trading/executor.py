"""
trading/executor.py — Full trade execution pipeline.
Handles auto-relax threshold management, market matching, Kelly sizing,
Kalshi API order placement, and Supabase logging.
"""
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Optional

import numpy as np

from config import (
    MIN_EDGE, ABSTENTION_BAND, AUTO_RELAX_INCREMENT, AUTO_RELAX_CAP,
    DRY_DAY_THRESHOLD, SPORTS,
)
from data.features import build_features_for_game, features_to_vector, SPORT_FEATURES
from models import xgboost_model, bayesian_model, meta_learner, calibration
from trading.kalshi_client import (
    search_sports_markets, get_implied_probability, get_balance,
    place_order, usd_to_contracts,
)
from trading.kelly import compute_kelly_bet
from db.supabase_client import (
    log_prediction, log_trade, log_threshold_event,
    get_consecutive_dry_days,
)


def _run_parallel_inference(
    xgb_model,
    bn_model,
    meta_model,
    calib_model,
    feature_dict: dict,
    feature_vec: np.ndarray,
    sport: str,
) -> tuple:
    """
    Run XGBoost and Bayesian Network inference in parallel.
    Returns (xgb_prob, bn_prob, final_prob).
    """
    results = {}

    def run_xgb():
        return xgboost_model.predict_proba(xgb_model, feature_vec)

    def run_bn():
        return bayesian_model.predict_proba(bn_model, feature_dict)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_xgb): "xgb",
            executor.submit(run_bn): "bn",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = 0.5

    xgb_prob = results.get("xgb", 0.5)
    bn_prob = results.get("bn", 0.5)

    # Stack and calibrate
    if meta_model is not None:
        raw_prob = meta_learner.predict(meta_model, xgb_prob, bn_prob)
    else:
        raw_prob = (xgb_prob + bn_prob) / 2.0

    if calib_model is not None:
        final_prob = calibration.calibrate(calib_model, raw_prob)
    else:
        final_prob = raw_prob

    return xgb_prob, bn_prob, final_prob


def get_effective_min_edge() -> float:
    """Check consecutive dry days and apply auto-relax if needed."""
    dry_days = get_consecutive_dry_days()
    if dry_days < DRY_DAY_THRESHOLD:
        return MIN_EDGE

    # Compute how many relax steps have been triggered
    extra_steps = dry_days - DRY_DAY_THRESHOLD + 1
    relaxed = MIN_EDGE - (extra_steps * AUTO_RELAX_INCREMENT)
    relaxed = max(relaxed, MIN_EDGE - AUTO_RELAX_CAP)
    return round(relaxed, 4)


def execute_for_sport(
    sport: str,
    games: list,
    volume_path: str = "/models",
    run_date: Optional[date] = None,
) -> dict:
    """
    Run full prediction and trading pipeline for one sport.
    games: list of game dicts from ESPN scraper.
    Returns dict summarizing trades placed and predictions logged.
    """
    if run_date is None:
        run_date = datetime.utcnow().date()

    summary = {"sport": sport, "games_evaluated": 0, "bets_placed": 0,
               "bets_skipped": 0, "errors": []}

    # Load models
    xgb_m = xgboost_model.load_model(sport, volume_path)
    bn_m = bayesian_model.load_model(sport, volume_path)
    meta_m = meta_learner.load_model(sport, volume_path)
    calib_m = calibration.load_model(sport, volume_path)

    if xgb_m is None or bn_m is None:
        summary["errors"].append(f"Models not found for {sport} — skipping")
        return summary

    # Get effective MIN_EDGE with auto-relax
    eff_min_edge = get_effective_min_edge()
    if eff_min_edge < MIN_EDGE:
        log_threshold_event(
            event_type="auto_relax",
            old_value=MIN_EDGE,
            new_value=eff_min_edge,
            reason=f"No trades for {get_consecutive_dry_days()} consecutive days",
            event_date=run_date,
        )

    # Account balance
    try:
        bankroll = get_balance()
        if bankroll <= 0:
            bankroll = 500.0  # default demo bankroll
    except Exception:
        bankroll = 500.0

    for game in games:
        summary["games_evaluated"] += 1
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        home_id = game.get("home_id", "")
        away_id = game.get("away_id", "")
        game_id = game.get("id", "")

        try:
            # Build features
            feature_dict = build_features_for_game(
                sport, home_team, away_team,
                game_meta={"home_id": home_id, "away_id": away_id},
            )
            feature_vec = features_to_vector(feature_dict, sport)

            # Parallel inference
            xgb_prob, bn_prob, final_prob = _run_parallel_inference(
                xgb_m, bn_m, meta_m, calib_m, feature_dict, feature_vec, sport
            )

            # Find matching Kalshi market
            kalshi_markets = search_sports_markets(home_team, away_team, sport)
            if not kalshi_markets:
                pred_id = log_prediction(
                    sport=sport, game_id=game_id,
                    home_team=home_team, away_team=away_team,
                    game_date=run_date,
                    xgb_prob=xgb_prob, bn_prob=bn_prob,
                    final_prob=final_prob,
                    kalshi_implied=0.5, edge=0.0,
                    decision="skip_no_market",
                )
                summary["bets_skipped"] += 1
                continue

            market = kalshi_markets[0]
            kalshi_implied = get_implied_probability(market)
            kalshi_yes_price = int(market.get("yes_price", 50) or 50)

            # Kelly sizing
            kelly_result = compute_kelly_bet(
                model_prob=final_prob,
                kalshi_yes_price=kalshi_yes_price,
                bankroll=bankroll,
                min_edge=eff_min_edge,
            )

            decision = "bet" if kelly_result["should_bet"] else "skip"
            edge = kelly_result["edge"]

            pred_id = log_prediction(
                sport=sport, game_id=game_id,
                home_team=home_team, away_team=away_team,
                game_date=run_date,
                xgb_prob=xgb_prob, bn_prob=bn_prob,
                final_prob=final_prob,
                kalshi_implied=kalshi_implied, edge=edge,
                decision=decision,
            )

            if not kelly_result["should_bet"]:
                summary["bets_skipped"] += 1
                continue

            # Place trade
            side = kelly_result["side"]
            bet_size = kelly_result["bet_size_usd"]
            n_contracts = usd_to_contracts(bet_size, kalshi_yes_price if side == "yes" else 100 - kalshi_yes_price)

            order_result = place_order(
                market_ticker=market.get("ticker", ""),
                side=side,
                count=n_contracts,
                price=kalshi_yes_price if side == "yes" else 100 - kalshi_yes_price,
            )

            if order_result is not None:
                log_trade(
                    prediction_id=pred_id,
                    kalshi_market_id=market.get("ticker", ""),
                    side=side,
                    bet_size_usd=bet_size,
                    kalshi_price=float(kalshi_yes_price if side == "yes" else 100 - kalshi_yes_price),
                    status="open",
                )
                summary["bets_placed"] += 1
            else:
                summary["bets_skipped"] += 1

        except Exception as exc:
            err_msg = str(exc)
            tb = traceback.format_exc()
            summary["errors"].append(f"{home_team} vs {away_team}: {err_msg}")
            from db.supabase_client import log_error
            log_error(
                context=f"executor({sport}) {home_team} vs {away_team}",
                error_msg=err_msg,
                tb=tb,
                run_date=run_date,
            )

    return summary
