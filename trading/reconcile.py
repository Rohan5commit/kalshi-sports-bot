"""
trading/reconcile.py — Standalone trade reconciliation logic.
Imported by both modal_app.py and email_report.py (no circular dependency).
"""
from datetime import datetime, timedelta


def reconcile_open_trades():
    """
    For each open trade in Supabase:
    - If market settled/closed/finalized: mark won or lost based on result vs side
    - If market active AND we have a position: keep open (order filled, awaiting settlement)
    - If market active AND no position AND game window passed (>3h after close_time): cancel
    - If market active AND no position AND game not started yet: keep open (GTC resting order)
    """
    from trading.kalshi_client import _public_request, get_open_positions
    from db.supabase_client import get_open_trades, update_trade_status

    open_trades = get_open_trades()
    if not open_trades:
        print("[reconcile] No open trades to reconcile")
        return

    positions = get_open_positions()
    held_tickers = {
        p.get("market_id", p.get("ticker", ""))
        for p in positions
        if int(p.get("yes_position", 0) or 0) > 0 or int(p.get("no_position", 0) or 0) > 0
    }

    now_utc = datetime.utcnow()
    print(f"[reconcile] Checking {len(open_trades)} open trade(s), {len(held_tickers)} filled position(s)...")
    for trade in open_trades:
        trade_id = trade.get("id")
        market_id = trade.get("kalshi_market_id", "")
        side = trade.get("side", "yes")
        if not market_id:
            continue

        try:
            market_data = _public_request("GET", f"/markets/{market_id}")
            if market_data is None:
                print(f"[reconcile] {market_id}: could not fetch — skipping")
                continue

            market = market_data.get("market", market_data)
            status = market.get("status", "")
            result = market.get("result", "")

            if status in ("settled", "closed", "finalized"):
                if result:
                    won = (side == "yes" and result == "yes") or (side == "no" and result == "no")
                    new_status = "won" if won else "lost"
                    update_trade_status(trade_id, new_status)
                    print(f"[reconcile] {market_id}: result={result} side={side} -> {new_status}")
                else:
                    if market_id in held_tickers:
                        update_trade_status(trade_id, "won")
                        print(f"[reconcile] {market_id}: {status} no-result but have position -> won")
                    else:
                        update_trade_status(trade_id, "lost")
                        print(f"[reconcile] {market_id}: {status} no-result no position -> lost")
                continue

            if market_id in held_tickers:
                print(f"[reconcile] {market_id}: active, position held — keeping open")
                continue

            close_time_str = market.get("close_time", "") or market.get("expiration_time", "")
            if close_time_str:
                try:
                    close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    if now_utc < close_time + timedelta(hours=3):
                        print(f"[reconcile] {market_id}: active, no fill yet, game not started — keeping open")
                        continue
                except (ValueError, TypeError):
                    pass

            update_trade_status(trade_id, "canceled")
            print(f"[reconcile] {market_id}: active, no contracts filled after game window -> canceled")

        except Exception as exc:
            print(f"[reconcile] {market_id}: error — {exc}")

