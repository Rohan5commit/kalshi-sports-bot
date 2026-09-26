"""
trading/kelly.py — Kelly criterion bet sizing for Kalshi prediction markets.
Uses quarter-Kelly by default; applies edge threshold guard only.
Abstention band removed: we use Vegas consensus probability, not uncertain ML output.
"""
from config import (
    KELLY_FRACTION,
    MAX_BET_USD,
    MIN_EDGE,
)


def compute_kelly_bet(
    model_prob: float,
    kalshi_yes_price: float,
    bankroll: float,
    min_edge: float = MIN_EDGE,
) -> dict:
    """
    Compute the optimal bet size using fractional Kelly criterion.

    Picks the side (YES or NO) with positive edge vs the Kalshi implied probability.
    Since signal is Vegas consensus, near-50% games are equally valid — no abstention band.

    Args:
        model_prob: vig-removed Vegas probability that the YES outcome wins (0-1)
        kalshi_yes_price: Kalshi yes price in cents (e.g., 60 means $0.60/contract)
        bankroll: available bankroll in USD
        min_edge: minimum edge required to place a bet

    Returns:
        dict with keys:
            should_bet (bool), bet_size_usd (float), edge (float),
            side (str: yes/no), reason (str)
    """
    if kalshi_yes_price <= 0 or kalshi_yes_price >= 100:
        return {"should_bet": False, "bet_size_usd": 0.0, "edge": 0.0,
                "side": "yes", "reason": "invalid_price"}

    implied_prob = kalshi_yes_price / 100.0
    no_implied_prob = 1.0 - implied_prob

    yes_edge = model_prob - implied_prob
    no_edge = (1.0 - model_prob) - no_implied_prob  # equiv: implied_prob - model_prob

    # Pick the side with positive edge
    if yes_edge >= no_edge:
        side = "yes"
        edge = yes_edge
        p_win = model_prob
        p_lose = 1.0 - model_prob
        odds = (1.0 - implied_prob) / implied_prob
    else:
        side = "no"
        edge = no_edge
        no_price = 1.0 - implied_prob
        p_win = 1.0 - model_prob
        p_lose = model_prob
        odds = implied_prob / no_price

    # Minimum edge filter
    if edge < min_edge:
        return {
            "should_bet": False,
            "bet_size_usd": 0.0,
            "edge": edge,
            "side": side,
            "reason": f"insufficient_edge ({edge:.4f} < {min_edge:.4f})",
        }

    # Kelly formula: f* = (b*p - q) / b
    if odds <= 0:
        return {
            "should_bet": False,
            "bet_size_usd": 0.0,
            "edge": edge,
            "side": side,
            "reason": "invalid_odds",
        }

    kelly_f = (odds * p_win - p_lose) / odds
    kelly_f = max(kelly_f, 0.0)

    # Apply fractional Kelly
    bet_fraction = KELLY_FRACTION * kelly_f
    bet_size = min(bet_fraction * bankroll, MAX_BET_USD)
    bet_size = max(bet_size, 0.0)

    if bet_size < 1.0:
        return {
            "should_bet": False,
            "bet_size_usd": 0.0,
            "edge": edge,
            "side": side,
            "reason": "bet_size_too_small",
        }

    return {
        "should_bet": True,
        "bet_size_usd": round(bet_size, 2),
        "edge": edge,
        "side": side,
        "reason": "ok",
    }


def compute_pnl(trades: list) -> float:
    """
    Compute running P&L from a list of trade dicts.
    trade: {status, bet_size_usd, kalshi_price, side}
    """
    pnl = 0.0
    for t in trades:
        if t.get("status") == "won":
            price = t.get("kalshi_price", 50) / 100.0
            size = t.get("bet_size_usd", 0.0)
            payout = size / price
            pnl += payout - size
        elif t.get("status") == "lost":
            pnl -= t.get("bet_size_usd", 0.0)
    return round(pnl, 2)
