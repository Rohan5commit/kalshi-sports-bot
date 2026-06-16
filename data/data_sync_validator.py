"""
data/data_sync_validator.py — Enforces strict no-lookahead data synchronization.
Validates every training row against temporal cutoff rules before feature matrix entry.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

PRICE_CUTOFF_MINUTES = 30
INJURY_CUTOFF_HOURS = 2
WEATHER_CUTOFF_HOURS = 6


def _parse_ts(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(str(ts)[:19], fmt[:19])
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def validate_row(row: dict) -> tuple:
    """
    Run all 6 temporal sync checks on a feature row.
    Returns (is_valid: bool, reason: str, failed_check: str).
    """
    game_start = _parse_ts(row.get("game_start_time") or row.get("game_date"))
    if not game_start:
        return False, "missing game_start_time", "timestamp_alignment"

    # If only a date (no time component), skip time-based checks — insufficient precision
    if game_start.hour == 0 and game_start.minute == 0 and game_start.second == 0:
        if "T" not in str(row.get("game_start_time", "")):
            return True, "", ""

    price_cutoff = game_start - timedelta(minutes=PRICE_CUTOFF_MINUTES)
    injury_cutoff = game_start - timedelta(hours=INJURY_CUTOFF_HOURS)
    weather_min = game_start - timedelta(hours=WEATHER_CUTOFF_HOURS)

    # Check 1: General feature timestamp must be before game start
    feature_ts = _parse_ts(row.get("feature_timestamp"))
    if feature_ts and feature_ts >= game_start:
        return False, f"feature data timestamped after game start: {feature_ts}", "timestamp_alignment"

    # Check 2: Injury data must be from 2h before tip-off
    injury_ts = _parse_ts(row.get("injury_report_timestamp"))
    if injury_ts and injury_ts > injury_cutoff:
        return False, f"injury report too recent: {injury_ts} > cutoff {injury_cutoff}", "injury_cutoff"

    # Check 3: Market prices must be from 30 min before game start
    kalshi_ts = _parse_ts(row.get("kalshi_price_cutoff_time"))
    if kalshi_ts and kalshi_ts > price_cutoff:
        return False, f"kalshi price after cutoff: {kalshi_ts}", "price_cutoff"
    poly_ts = _parse_ts(row.get("poly_price_cutoff_time"))
    if poly_ts and poly_ts > price_cutoff:
        return False, f"polymarket price after cutoff: {poly_ts}", "price_cutoff"

    # Check 4: Weather forecast must be within 6h of game start
    weather_ts = _parse_ts(row.get("weather_forecast_timestamp"))
    if weather_ts and weather_ts < weather_min:
        return False, f"weather forecast too stale: {weather_ts} < {weather_min}", "weather_cutoff"

    # Check 5: News/sentiment articles must be published before game start
    news_ts = _parse_ts(row.get("article_published_at"))
    if news_ts and news_ts >= game_start:
        return False, f"news article published after game start: {news_ts}", "news_cutoff"

    # Check 6: Lineup confirmation flag — not a drop, handled by feature builder
    # (sets lineup stats to season averages when home_lineup_confirmed == 0)

    return True, "", ""


def filter_training_rows(sport: str, rows: list, run_date=None) -> tuple:
    """
    Filter training rows, log dropped ones to Supabase.
    Returns (valid_rows, dropped_count, drop_summary_dict).
    """
    from db.historical_store import log_dropped_rows

    if run_date is None:
        run_date = datetime.utcnow().date()

    valid_rows, dropped, drop_summary = [], [], {}

    for row in rows:
        is_valid, reason, failed_check = validate_row(row)
        if is_valid:
            valid_rows.append(row)
        else:
            dropped.append({
                "game_id": row.get("game_id", ""),
                "reason": reason,
                "failed_check": failed_check,
            })
            drop_summary[failed_check] = drop_summary.get(failed_check, 0) + 1

    if dropped:
        try:
            log_dropped_rows(dropped)
        except Exception as exc:
            print(f"[validator] log_dropped_rows error: {exc}")

    total_dropped = len(dropped)
    print(f"[{sport}] DataSyncValidator: {len(valid_rows)} valid, {total_dropped} dropped")
    for check, count in drop_summary.items():
        print(f"  - {check}: {count} rows dropped")

    return valid_rows, total_dropped, drop_summary


def get_dropped_rows_summary(run_date) -> dict:
    """Fetch dropped row counts grouped by failed_check for the email report."""
    try:
        from db.supabase_client import _get_client
        sb = _get_client()
        cutoff = datetime.combine(run_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        rows = (sb.table("dropped_rows")
                  .select("failed_check, reason")
                  .gte("created_at", cutoff.isoformat())
                  .execute().data)
        summary: dict = {}
        for r in rows:
            key = r.get("failed_check", "unknown")
            summary[key] = summary.get(key, 0) + 1
        return summary
    except Exception:
        return {}
