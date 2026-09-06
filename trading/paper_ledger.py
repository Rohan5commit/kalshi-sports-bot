"""
trading/paper_ledger.py — Paper trading ledger: bankroll tracking and paper trade persistence.
All data stored in Supabase paper_bankroll and paper_trades tables.
"""
from datetime import datetime, date
from typing import Optional

from db.supabase_client import _get_client, _retry
from config import PAPER_BANKROLL_START


def get_bankroll() -> float:
    """
    Read current bankroll from paper_bankroll table.
    Falls back to PAPER_BANKROLL_START from config if table empty or error.
    """
    def _fetch():
        sb = _get_client()
        res = sb.table("paper_bankroll").select("bankroll").limit(1).execute()
        if res.data:
            return float(res.data[0]["bankroll"])
        return None

    try:
        val = _retry(_fetch)
        if val is not None:
            return val
    except Exception as exc:
        print(f"[paper_ledger] get_bankroll error: {exc}")
    return PAPER_BANKROLL_START


def update_bankroll(new_value: float, delta_pnl: float = 0.0) -> None:
    """
    Update the bankroll row. If no row exists, inserts one.
    Adds delta_pnl to total_pnl accumulator.
    """
    def _update():
        sb = _get_client()
        res = sb.table("paper_bankroll").select("id,total_pnl").limit(1).execute()
        if res.data:
            row = res.data[0]
            current_pnl = float(row.get("total_pnl") or 0.0)
            sb.table("paper_bankroll").update({
                "bankroll": round(new_value, 2),
                "total_pnl": round(current_pnl + delta_pnl, 2),
                "last_updated": datetime.utcnow().isoformat(),
            }).eq("id", row["id"]).execute()
        else:
            sb.table("paper_bankroll").insert({
                "bankroll": round(new_value, 2),
                "total_pnl": round(delta_pnl, 2),
            }).execute()

    try:
        _retry(_update)
    except Exception as exc:
        print(f"[paper_ledger] update_bankroll error: {exc}")


def log_paper_trade(
    kalshi_market_id: str,
    side: str,
    count: int,
    fill_price: float,
    bet_usd: float,
    fee: float,
    model_prob: float,
    market_prob: float,
    net_edge: float,
    sport: Optional[str] = None,
    game_date=None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    run_date=None,
) -> str:
    """
    Insert a paper trade record into paper_trades table.
    Returns the UUID string of the inserted row.
    """
    def _insert():
        sb = _get_client()
        row = {
            "kalshi_market_id": kalshi_market_id,
            "side": side,
            "count": count,
            "fill_price": round(float(fill_price), 4),
            "bet_usd": round(float(bet_usd), 2),
            "fee": round(float(fee), 4),
            "model_prob": round(float(model_prob), 4) if model_prob is not None else None,
            "market_prob": round(float(market_prob), 4) if market_prob is not None else None,
            "net_edge": round(float(net_edge), 4) if net_edge is not None else None,
            "status": "open",
        }
        if sport is not None:
            row["sport"] = sport
        if game_date is not None:
            row["game_date"] = str(game_date)
        if home_team is not None:
            row["home_team"] = home_team
        if away_team is not None:
            row["away_team"] = away_team
        if run_date is not None:
            row["run_date"] = str(run_date)
        res = sb.table("paper_trades").insert(row).execute()
        return res.data[0]["id"]

    return _retry(_insert)


def get_open_paper_trades() -> list:
    """Return all paper trades with status='open'."""
    def _fetch():
        sb = _get_client()
        res = sb.table("paper_trades").select("*").eq("status", "open").execute()
        return res.data or []

    try:
        return _retry(_fetch)
    except Exception as exc:
        print(f"[paper_ledger] get_open_paper_trades error: {exc}")
        return []


def settle_paper_trade(trade_id: str, result: str, pnl: float) -> None:
    """
    Settle a paper trade.
    result: 'won' or 'lost'
    Sets status, settlement_pnl, and settled_at timestamp.
    """
    def _update():
        sb = _get_client()
        sb.table("paper_trades").update({
            "status": result,
            "settlement_pnl": round(float(pnl), 2),
            "settled_at": datetime.utcnow().isoformat(),
        }).eq("id", trade_id).execute()

    try:
        _retry(_update)
    except Exception as exc:
        print(f"[paper_ledger] settle_paper_trade error: {exc}")


def open_exposure() -> float:
    """
    Calculate total open exposure across all open paper trades.
    Exposure = sum of (fill_price * count + fee) for each open trade.
    """
    def _fetch():
        sb = _get_client()
        res = sb.table("paper_trades").select("fill_price,count,fee").eq("status", "open").execute()
        return res.data or []

    try:
        trades = _retry(_fetch)
        total = 0.0
        for t in trades:
            fill_price = float(t.get("fill_price", 0))
            count = int(t.get("count", 0))
            fee = float(t.get("fee", 0))
            total += fill_price * count + fee
        return round(total, 4)
    except Exception as exc:
        print(f"[paper_ledger] open_exposure error: {exc}")
        return 0.0
