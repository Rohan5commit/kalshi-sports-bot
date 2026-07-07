"""
db/supabase_client.py — Supabase client using supabase-py REST API.
No direct postgres connection needed — works from any cloud environment.
"""
import os
import traceback
import time
from datetime import datetime, date, timedelta
from typing import Optional

from supabase import create_client, Client

from config import MAX_RETRIES, BACKOFF_BASE


def _get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def _retry(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** attempt)
    raise last_exc


# ── Table creation (via raw SQL through RPC) ──────────────────────────────────

def create_tables():
    """Tables are pre-created via Supabase MCP migration — this is a no-op check."""
    try:
        sb = _get_client()
        sb.table("predictions").select("id").limit(1).execute()
        print("Supabase tables confirmed reachable")
    except Exception as exc:
        print(f"Warning: could not verify tables: {exc}")


# ── Insert helpers ─────────────────────────────────────────────────────────────

def log_prediction(sport: str, game_id: str, home_team: str, away_team: str,
                   game_date: date, xgb_prob: float, bn_prob: float,
                   final_prob: float, kalshi_implied: float, edge: float,
                   decision: str) -> str:
    def _insert():
        sb = _get_client()
        res = sb.table("predictions").insert({
            "sport": sport, "game_id": game_id,
            "home_team": home_team, "away_team": away_team,
            "game_date": str(game_date),
            "xgb_prob": xgb_prob, "bn_prob": bn_prob,
            "final_prob": final_prob, "kalshi_implied": kalshi_implied,
            "edge": edge, "decision": decision,
        }).execute()
        return res.data[0]["id"]
    return _retry(_insert)


def log_trade(prediction_id: str, kalshi_market_id: str, side: str,
              bet_size_usd: float, kalshi_price: float, status: str = "open") -> str:
    def _insert():
        sb = _get_client()
        res = sb.table("trades").insert({
            "prediction_id": prediction_id,
            "kalshi_market_id": kalshi_market_id,
            "side": side, "bet_size_usd": bet_size_usd,
            "kalshi_price": kalshi_price, "status": status,
        }).execute()
        return res.data[0]["id"]
    return _retry(_insert)


def log_model_performance(run_date: date, sport: str, accuracy_14d: float,
                           total_predictions: int, retrain_triggered: bool):
    def _insert():
        sb = _get_client()
        sb.table("model_performance").insert({
            "date": str(run_date), "sport": sport,
            "accuracy_14d": accuracy_14d,
            "total_predictions": total_predictions,
            "retrain_triggered": retrain_triggered,
        }).execute()
    _retry(_insert)


def log_error(context: str, error_msg: str, tb: str, run_date: Optional[date] = None):
    if run_date is None:
        run_date = datetime.utcnow().date()
    def _insert():
        sb = _get_client()
        sb.table("errors").insert({
            "run_date": str(run_date),
            "context": context, "error_msg": error_msg, "traceback": tb,
        }).execute()
    try:
        _retry(_insert)
    except Exception:
        pass  # never crash when logging errors


def log_threshold_event(event_type: str, old_value: float, new_value: float,
                        reason: str, event_date: Optional[date] = None):
    if event_date is None:
        event_date = datetime.utcnow().date()
    def _insert():
        sb = _get_client()
        sb.table("threshold_events").insert({
            "event_date": str(event_date),
            "event_type": event_type,
            "old_value": old_value, "new_value": new_value, "reason": reason,
        }).execute()
    _retry(_insert)


# ── Query helpers ──────────────────────────────────────────────────────────────

def get_todays_predictions(run_date: date) -> list:
    def _query():
        sb = _get_client()
        res = sb.table("predictions").select("*").eq("game_date", str(run_date)).execute()
        return res.data
    return _retry(_query)


def get_todays_trades(run_date: date) -> list:
    def _query():
        sb = _get_client()
        preds = sb.table("predictions").select("id,sport,home_team,away_team,final_prob,edge").eq("game_date", str(run_date)).execute()
        pred_ids = [p["id"] for p in preds.data]
        if not pred_ids:
            return []
        trades = sb.table("trades").select("*").in_("prediction_id", pred_ids).execute()
        pred_map = {p["id"]: p for p in preds.data}
        result = []
        for t in trades.data:
            p = pred_map.get(t["prediction_id"], {})
            result.append({**t, "sport": p.get("sport"), "home_team": p.get("home_team"),
                           "away_team": p.get("away_team"), "final_prob": p.get("final_prob"),
                           "edge": p.get("edge")})
        return result
    return _retry(_query)


def get_open_trades() -> list:
    def _query():
        sb = _get_client()
        trades = sb.table("trades").select("*").eq("status", "open").execute()
        result = []
        for t in trades.data:
            pred = sb.table("predictions").select("sport,home_team,away_team").eq("id", t["prediction_id"]).maybe_single().execute()
            p = pred.data or {}
            result.append({**t, "sport": p.get("sport"), "home_team": p.get("home_team"),
                           "away_team": p.get("away_team")})
        return result
    return _retry(_query)


def get_todays_errors(run_date: date) -> list:
    def _query():
        sb = _get_client()
        res = sb.table("errors").select("*").eq("run_date", str(run_date)).execute()
        return res.data
    return _retry(_query)


def get_todays_threshold_events(run_date: date) -> list:
    def _query():
        sb = _get_client()
        res = sb.table("threshold_events").select("*").eq("event_date", str(run_date)).execute()
        return res.data
    return _retry(_query)


def get_resolved_games_since(sport: str, since_date: date) -> list:
    def _query():
        sb = _get_client()
        res = (sb.table("predictions")
                 .select("*")
                 .eq("sport", sport)
                 .gte("game_date", str(since_date))
                 .execute())
        return res.data
    return _retry(_query)


def update_trade_status(trade_id: str, status: str):
    def _update():
        sb = _get_client()
        sb.table("trades").update({"status": status}).eq("id", trade_id).execute()
    _retry(_update)


def get_closed_trades() -> list:
    """Return all trades with status won or lost, enriched with prediction data."""
    def _query():
        sb = _get_client()
        trades = sb.table("trades").select("*").in_("status", ["won","lost"]).order("created_at", desc=True).execute()
        result = []
        for t in trades.data:
            pred = sb.table("predictions").select("sport,home_team,away_team").eq("id", t["prediction_id"]).maybe_single().execute()
            p = pred.data or {}
            result.append({**t, "sport": p.get("sport"), "home_team": p.get("home_team"),
                           "away_team": p.get("away_team")})
        return result
    return _retry(_query)


def get_consecutive_dry_days() -> int:
    """Count consecutive days with no trades placed. Uses trades table (more reliable than predictions)."""
    def _query():
        sb = _get_client()
        # Use trades table — it writes even when predictions fail (e.g. during RLS issues)
        res = (sb.table("trades")
                 .select("created_at")
                 .not_.in_("status", ["canceled"])
                 .order("created_at", desc=True)
                 .limit(30)
                 .execute())
        return res.data
    rows = _retry(_query)
    # Extract just the date portion from created_at timestamps
    dates_with_trades = {r["created_at"][:10] for r in rows if r.get("created_at")}
    today = datetime.utcnow().date()
    dry = 0
    d = today
    while str(d) not in dates_with_trades:
        dry += 1
        d -= timedelta(days=1)
        if dry > 30:
            break
    return dry
