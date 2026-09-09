"""
trading/executor.py — Full trade execution pipeline.
Primary signal: Vegas moneyline (DraftKings via ESPN) converted to vig-removed probability.
ML models used as secondary signal when Vegas odds unavailable.

Paper trading mode: bets are simulated against live orderbook snapshots.
No real orders are placed. Fills tracked in Supabase paper_trades table.
"""
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Optional

import numpy as np

from config import (
    MIN_EDGE, ABSTENTION_BAND, AUTO_RELAX_INCREMENT, AUTO_RELAX_CAP,
    DRY_DAY_THRESHOLD, SPORTS,
    MAX_BET_PCT, MIN_BET_USD, LOSS_FLOOR_PCT, ORDERBOOK_DEPTH, PAPER_BANKROLL_START,
)
from data.features import build_features_for_game, features_to_vector, SPORT_FEATURES
from models import xgboost_model, bayesian_model, meta_learner, calibration
from trading.kalshi_client import (
    search_sports_markets, get_implied_probability, get_yes_price_cents,
    usd_to_contracts, parse_bundle_legs,
)
from trading.shadow_book import get_orderbook, simulate_fill, calc_fee
from trading.paper_ledger import (
    log_paper_trade, update_bankroll, get_bankroll as get_paper_bankroll, open_exposure,
)
from trading.kelly import compute_kelly_bet
from db.supabase_client import (
    log_prediction, log_trade, log_threshold_event,
    get_consecutive_dry_days, get_open_trades, get_todays_threshold_events,
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
    dry_days = get_consecutive_dry_days()
    if dry_days < DRY_DAY_THRESHOLD:
        return MIN_EDGE
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
    Run full prediction and paper trading pipeline for one sport.

    Signal priority:
    1. Vegas moneyline (DraftKings via ESPN) — calibrated, real consensus probability
    2. ML models (XGBoost + BN) — used only when Vegas odds absent

    Edge = |vegas_prob - kalshi_implied|. We bet when Kalshi misprices vs Vegas.
    Paper trading: fills are simulated against live orderbook; no real orders placed.
    """
    if run_date is None:
        run_date = datetime.utcnow().date()

    summary = {"sport": sport, "games_evaluated": 0, "bets_placed": 0,
               "bets_skipped": 0, "errors": []}

    # Load ML models (optional fallback — won't block if missing)
    xgb_m = xgboost_model.load_model(sport, volume_path)
    bn_m = bayesian_model.load_model(sport, volume_path)
    meta_m = meta_learner.load_model(sport, volume_path)
    calib_m = calibration.load_model(sport, volume_path)

    eff_min_edge = get_effective_min_edge()
    if eff_min_edge < MIN_EDGE:
        try:
            existing = get_todays_threshold_events(run_date)
            if not any(e.get("event_type") == "auto_relax" for e in existing):
                log_threshold_event(
                    event_type="auto_relax",
                    old_value=MIN_EDGE,
                    new_value=eff_min_edge,
                    reason=f"No trades for {get_consecutive_dry_days()} consecutive days",
                    event_date=run_date,
                )
        except Exception:
            pass

    # Use paper bankroll instead of live Kalshi balance
    bankroll = get_paper_bankroll()
    if bankroll <= 0:
        bankroll = PAPER_BANKROLL_START

    # Build set of market tickers already open/resting — don't double-bet same game
    already_open = {t.get("kalshi_market_id") for t in get_open_trades()}

    for game in games:
        summary["games_evaluated"] += 1
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        home_id = game.get("home_id", "")
        away_id = game.get("away_id", "")
        home_abbr = game.get("home_abbr", "")
        away_abbr = game.get("away_abbr", "")
        game_id = game.get("id", "")
        vegas_home_prob = game.get("vegas_home_prob")  # vig-removed, from ESPN/DraftKings
        game_label = f"{away_team} @ {home_team}"

        try:
            # Find Kalshi market
            kalshi_markets = search_sports_markets(home_team, away_team, sport,
                                                   home_abbr=home_abbr, away_abbr=away_abbr)
            live_kalshi_price = 0.5
            if kalshi_markets:
                raw_cents = get_yes_price_cents(kalshi_markets[0])
                if raw_cents == 0:
                    kalshi_markets = []
                else:
                    live_kalshi_price = raw_cents / 100.0

            # Vegas moneyline is the only signal we trust.
            # ML models are untrained and produce noise — skip when no Vegas line.
            xgb_prob = bn_prob = 0.5
            if vegas_home_prob is None:
                summary["bets_skipped"] += 1
                print(f"  {game_label}: skip (no Vegas odds available)")
                continue
            final_prob = vegas_home_prob
            model_prob = final_prob
            signal_source = f"vegas({vegas_home_prob:.3f})"

            if not kalshi_markets:
                log_prediction(
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
            market_ticker = market.get("ticker", "")

            if market_ticker in already_open:
                print(f"  {game_label}: skip (already have open order)")
                summary["bets_skipped"] += 1
                continue
            is_bundle = bool(market.get("_is_bundle"))
            kalshi_implied = get_implied_probability(market)
            kalshi_yes_price = get_yes_price_cents(market)

            if is_bundle and kalshi_yes_price > 0:
                bundle_legs = parse_bundle_legs(market.get("title", ""))
                n_legs = max(len(bundle_legs), 1)
                bundle_price = kalshi_yes_price / 100.0
                implied_per_leg = bundle_price ** (1.0 / n_legs)
                effective_model_prob = max(0.0, min(1.0,
                    (final_prob / implied_per_leg) * bundle_price if implied_per_leg > 0 else bundle_price
                ))
                effective_min_edge = eff_min_edge * 2.0
            else:
                effective_model_prob = final_prob
                effective_min_edge = eff_min_edge

            kelly_result = compute_kelly_bet(
                model_prob=effective_model_prob,
                kalshi_yes_price=kalshi_yes_price,
                bankroll=bankroll,
                min_edge=effective_min_edge,
            )

            decision = (
                "bet_bundle" if (kelly_result["should_bet"] and is_bundle) else
                "bet" if kelly_result["should_bet"] else
                "skip"
            )
            edge = kelly_result["edge"]
            print(f"  {game_label}: {signal_source} kalshi={kalshi_implied:.2f} edge={edge:+.3f} → {decision} ({kelly_result['reason']})")

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

            side = kelly_result["side"]
            bet_size = kelly_result["bet_size_usd"]

            # ── Paper fill pipeline ────────────────────────────────────────────

            # 1. Check effective loss floor
            exposure = open_exposure()
            effective_bankroll = bankroll - exposure
            loss_pct = (PAPER_BANKROLL_START - effective_bankroll) / PAPER_BANKROLL_START
            if loss_pct >= LOSS_FLOOR_PCT:
                print(f"  {game_label}: HALT — loss floor breached (effective=${effective_bankroll:.2f})")
                summary["bets_skipped"] += 1
                continue

            # 2. Fetch live orderbook from production
            book = get_orderbook(market_ticker, depth=ORDERBOOK_DEPTH, force_resync=True)
            side_levels = book.get("yes_levels" if side == "yes" else "no_levels", [])

            if not side_levels:
                print(f"  {game_label}: skip — no orderbook depth for {market_ticker}")
                summary["bets_skipped"] += 1
                continue

            ask_price = side_levels[0][0]  # best ask (ascending sort)

            # 3. Phase 1 edge check with fee at best ask
            # Edge formula is side-aware: YES edge = model_prob - ask; NO edge = (1-model_prob) - ask
            fee_per = calc_fee(ask_price, 1)
            if side == "yes":
                net_edge_p1 = model_prob - ask_price - fee_per
            else:
                net_edge_p1 = (1.0 - model_prob) - ask_price - fee_per
            if net_edge_p1 < eff_min_edge:
                print(f"  {game_label}: skip — edge evaporated at ask (net={net_edge_p1:.4f})")
                summary["bets_skipped"] += 1
                continue

            # 4. Compute contract count from kelly bet_size, cap at MAX_BET_PCT
            max_bet_usd = MAX_BET_PCT * bankroll
            capped_bet = min(bet_size, max_bet_usd)
            count = max(1, int(capped_bet / ask_price))

            # 5. Simulate fill
            filled, avg_fill = simulate_fill(side_levels, count, ask_price)
            if filled == 0:
                print(f"  {game_label}: skip — zero fill simulated")
                summary["bets_skipped"] += 1
                continue

            # 6. Phase 2 edge check at avg fill price (side-aware)
            total_fee = calc_fee(avg_fill, filled)
            if side == "yes":
                net_edge_p2 = model_prob - avg_fill - calc_fee(avg_fill, 1)
            else:
                net_edge_p2 = (1.0 - model_prob) - avg_fill - calc_fee(avg_fill, 1)
            if net_edge_p2 < eff_min_edge:
                print(f"  {game_label}: skip — edge evaporated at fill (net={net_edge_p2:.4f}, avg={avg_fill:.4f})")
                summary["bets_skipped"] += 1
                continue

            # 7. Check minimum bet
            actual_bet_usd = filled * avg_fill
            if actual_bet_usd < MIN_BET_USD:
                print(f"  {game_label}: skip — below min bet (${actual_bet_usd:.2f})")
                summary["bets_skipped"] += 1
                continue

            # 8. Book the paper fill
            trade_id = log_paper_trade(
                kalshi_market_id=market_ticker,
                side=side,
                count=filled,
                fill_price=avg_fill,
                bet_usd=actual_bet_usd,
                fee=total_fee,
                model_prob=model_prob,
                market_prob=ask_price,
                net_edge=net_edge_p2,
                sport=sport,
                game_date=run_date,
                home_team=game.get("home_team"),
                away_team=game.get("away_team"),
                run_date=run_date,
            )

            # Also log to predictions/trades table for historical tracking
            log_trade(
                prediction_id=pred_id,
                kalshi_market_id=market_ticker,
                side=side,
                bet_size_usd=actual_bet_usd,
                kalshi_price=float(round(avg_fill * 100)),
                status="open",
            )

            # 9. Debit bankroll
            new_bankroll = bankroll - actual_bet_usd - total_fee
            update_bankroll(new_bankroll)
            bankroll = new_bankroll

            print(f"  {game_label}: PAPER FILL {filled}x @ {avg_fill:.4f} | net_edge={net_edge_p2:.4f} | bet=${actual_bet_usd:.2f} fee=${total_fee:.4f} | id={trade_id[:8]}")
            summary["bets_placed"] += 1

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

