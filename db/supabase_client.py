"""
db/supabase_client.py — Supabase client, table creation, and all logging helpers.
"""
import os
import traceback
import time
from datetime import datetime, date
from typing import Optional, Any
import uuid

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from config import MAX_RETRIES, BACKOFF_BASE


def _get_conn():
    """Return a new psycopg2 connection using env vars from Modal Secret."""
    url = os.environ["SUPABASE_DB_URL"]
    return psycopg2.connect(url, cursor_factory=RealDictCursor)


def _retry(fn, *args, **kwargs):
    """Call fn with retries and exponential backoff."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** attempt)
    raise last_exc


# ── Table creation ────────────────────────────────────────────────────────────

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz DEFAULT now(),
    sport text,
    game_id text,
    home_team text,
    away_team text,
    game_date date,
    xgb_prob float,
    bn_prob float,
    final_prob float,
    kalshi_implied float,
    edge float,
    decision text
);
"""

CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz DEFAULT now(),
    prediction_id uuid REFERENCES predictions(id),
    kalshi_market_id text,
    side text,
    bet_size_usd float,
    kalshi_price float,
    status text
);
"""

CREATE_MODEL_PERFORMANCE = """
CREATE TABLE IF NOT EXISTS model_performance (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    date date,
    sport text,
    accuracy_14d float,
    total_predictions int,
    retrain_triggered boolean
);
"""

CREATE_ERRORS = """
CREATE TABLE IF NOT EXISTS errors (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz DEFAULT now(),
    run_date date,
    context text,
    error_msg text,
    traceback text
);
"""

CREATE_THRESHOLD_EVENTS = """
CREATE TABLE IF NOT EXISTS threshold_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz DEFAULT now(),
    event_date date,
    event_type text,
    old_value float,
    new_value float,
    reason text
);
"""


def create_tables():
    """Create all required tables if they don't exist."""
    def _create():
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                for ddl in [CREATE_PREDICTIONS, CREATE_TRADES,
                            CREATE_MODEL_PERFORMANCE, CREATE_ERRORS,
                            CREATE_THRESHOLD_EVENTS]:
                    cur.execute(ddl)
        conn.close()
    _retry(_create)


# ── Insert helpers ─────────────────────────────────────────────────────────────

def log_prediction(sport: str, game_id: str, home_team: str, away_team: str,
                   game_date: date, xgb_prob: float, bn_prob: float,
                   final_prob: float, kalshi_implied: float, edge: float,
                   decision: str) -> str:
    """Insert a prediction row and return its UUID."""
    def _insert():
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO predictions
                       (sport, game_id, home_team, away_team, game_date,
                        xgb_prob, bn_prob, final_prob, kalshi_implied, edge, decision)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       RETURNING id""",
                    (sport, game_id, home_team, away_team, game_date,
                     xgb_prob, bn_prob, final_prob, kalshi_implied, edge, decision)
                )
                row = cur.fetchone()
        conn.close()
        return str(row["id"])
    return _retry(_insert)


def log_trade(prediction_id: str, kalshi_market_id: str, side: str,
              bet_size_usd: float, kalshi_price: float, status: str = "open") -> str:
    """Insert a trade row and return its UUID."""
    def _insert():
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO trades
                       (prediction_id, kalshi_market_id, side, bet_size_usd, kalshi_price, status)
                       VALUES (%s,%s,%s,%s,%s,%s)
                       RETURNING id""",
                    (prediction_id, kalshi_market_id, side,
                     bet_size_usd, kalshi_price, status)
                )
                row = cur.fetchone()
        conn.close()
        return str(row["id"])
    return _retry(_insert)


def log_model_performance(run_date: date, sport: str, accuracy_14d: float,
                           total_predictions: int, retrain_triggered: bool):
    """Insert a model_performance row."""
    def _insert():
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO model_performance
                       (date, sport, accuracy_14d, total_predictions, retrain_triggered)
                       VALUES (%s,%s,%s,%s,%s)""",
                    (run_date, sport, accuracy_14d, total_predictions, retrain_triggered)
                )
        conn.close()
    _retry(_insert)


def log_error(context: str, error_msg: str, tb: str, run_date: Optional[date] = None):
    """Insert an error row."""
    if run_date is None:
        run_date = datetime.utcnow().date()
    def _insert():
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO errors (run_date, context, error_msg, traceback)
                       VALUES (%s,%s,%s,%s)""",
                    (run_date, context, error_msg, tb)
                )
        conn.close()
    try:
        _retry(_insert)
    except Exception:
        pass  # never crash when logging errors


def log_threshold_event(event_type: str, old_value: float, new_value: float,
                        reason: str, event_date: Optional[date] = None):
    """Log an auto-relax or similar threshold change event."""
    if event_date is None:
        event_date = datetime.utcnow().date()
    def _insert():
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO threshold_events
                       (event_date, event_type, old_value, new_value, reason)
                       VALUES (%s,%s,%s,%s,%s)""",
                    (event_date, event_type, old_value, new_value, reason)
                )
        conn.close()
    _retry(_insert)


# ── Query helpers ──────────────────────────────────────────────────────────────

def get_todays_predictions(run_date: date) -> list:
    def _query():
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM predictions WHERE game_date = %s ORDER BY created_at", (run_date,))
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    return _retry(_query)


def get_todays_trades(run_date: date) -> list:
    def _query():
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT t.*, p.sport, p.home_team, p.away_team, p.final_prob, p.edge
                   FROM trades t
                   JOIN predictions p ON t.prediction_id = p.id
                   WHERE p.game_date = %s
                   ORDER BY t.created_at""",
                (run_date,)
            )
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    return _retry(_query)


def get_open_trades() -> list:
    def _query():
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT t.*, p.sport, p.home_team, p.away_team
                   FROM trades t
                   JOIN predictions p ON t.prediction_id = p.id
                   WHERE t.status = 'open'
                   ORDER BY t.created_at DESC""",
            )
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    return _retry(_query)


def get_todays_errors(run_date: date) -> list:
    def _query():
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM errors WHERE run_date = %s ORDER BY created_at", (run_date,))
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    return _retry(_query)


def get_todays_threshold_events(run_date: date) -> list:
    def _query():
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM threshold_events WHERE event_date = %s ORDER BY created_at",
                (run_date,)
            )
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    return _retry(_query)


def get_resolved_games_since(sport: str, since_date: date) -> list:
    """Return predictions with game results (for retraining)."""
    def _query():
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT p.*, t.status as trade_status
                   FROM predictions p
                   LEFT JOIN trades t ON t.prediction_id = p.id
                   WHERE p.sport = %s AND p.game_date >= %s
                   ORDER BY p.game_date""",
                (sport, since_date)
            )
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    return _retry(_query)


def update_trade_status(trade_id: str, status: str):
    """Update trade status (open/won/lost)."""
    def _update():
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE trades SET status = %s WHERE id = %s", (status, trade_id))
        conn.close()
    _retry(_update)


def get_consecutive_dry_days() -> int:
    """Return number of consecutive days (going back from today) with no trades placed."""
    def _query():
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT game_date, COUNT(*) as bets
                   FROM predictions
                   WHERE decision = 'bet' AND game_date <= CURRENT_DATE
                   GROUP BY game_date
                   ORDER BY game_date DESC
                   LIMIT 30"""
            )
            rows = cur.fetchall()
        conn.close()
        return rows
    rows = _retry(_query)
    dates_with_bets = {r["game_date"] for r in rows}
    today = datetime.utcnow().date()
    dry = 0
    from datetime import timedelta
    d = today
    while d not in dates_with_bets:
        dry += 1
        d -= timedelta(days=1)
        if dry > 30:
            break
    return dry
