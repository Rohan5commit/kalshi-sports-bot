"""
trading/shadow_book.py — Shadow order book for paper trade fill simulation.
Fetches real production Kalshi orderbooks (GET only — never POSTs orders).
Caches snapshots for 30s to avoid hammering the API.
"""
import math
import time
from typing import Optional

from trading.kalshi_client import _authed_request

# Module-level cache: {ticker: {"yes_levels": [...], "no_levels": [...], "last_sync": float, "stale": bool}}
_SHADOW_CACHE: dict = {}
_RESYNC_TIMEOUT = 30  # seconds


def calc_fee(price: float, count: int) -> float:
    """
    Calculate total Kalshi trading fee.
    fee_per_contract = ceil(0.07 * price * (1 - price) * 100) / 100
    total_fee = fee_per_contract * count
    price should be in [0, 1] (dollars, not cents).
    """
    fee_per = math.ceil(0.07 * price * (1.0 - price) * 100) / 100
    return round(fee_per * count, 4)


def _parse_levels(raw: list) -> list:
    """
    Parse raw orderbook levels into (price, count) tuples.
    Handles both [[price, count], ...] arrays and [{"price": x, "count": y}, ...] dicts.
    Normalizes cents -> dollars if price > 1.0.
    Rejects entries where price is outside (0, 1) after normalization.
    Returns levels sorted ascending by price (best ask first).
    """
    parsed = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            p, c = entry[0], entry[1]
        elif isinstance(entry, dict):
            p = entry.get("price", 0)
            c = entry.get("count", 0)
        else:
            continue
        try:
            p = float(p)
            c = int(c)
        except (TypeError, ValueError):
            continue
        # Normalize cents to dollars
        if p > 1.0:
            p = p / 100.0
        # Reject invalid prices
        if not (0 < p < 1):
            continue
        if c <= 0:
            continue
        parsed.append((p, c))
    parsed.sort(key=lambda x: x[0])
    return parsed


def get_orderbook(ticker: str, depth: int = 20, force_resync: bool = False) -> dict:
    """
    Fetch and cache an order book for a given Kalshi market ticker.

    Returns:
        {
            "yes_levels": [(price, count), ...],  # ascending (best ask first)
            "no_levels": [(price, count), ...],
            "last_sync": float,                    # epoch time of last successful fetch
            "stale": bool,                         # True if using cached data past TTL
        }

    On fetch failure: returns stale cache if available, else empty dict.
    Cache TTL: 30 seconds.
    """
    now = time.time()
    cached = _SHADOW_CACHE.get(ticker)
    
    # Use cache if fresh and no force resync
    if cached and not force_resync:
        age = now - cached.get("last_sync", 0)
        if age < _RESYNC_TIMEOUT:
            return cached

    # Attempt live fetch
    raw = _authed_request("GET", f"/markets/{ticker}/orderbook?depth={depth}")
    
    if raw is None:
        # Fetch failed — return stale cache if available, else empty dict
        if cached:
            stale = dict(cached)
            stale["stale"] = True
            return stale
        return {}

    # Handle both response shapes:
    # {"orderbook": {"yes": [...], "no": [...]}} or {"yes": [...], "no": [...]}
    ob = raw.get("orderbook", raw)
    yes_raw = ob.get("yes", [])
    no_raw = ob.get("no", [])

    result = {
        "yes_levels": _parse_levels(yes_raw),
        "no_levels": _parse_levels(no_raw),
        "last_sync": now,
        "stale": False,
    }
    _SHADOW_CACHE[ticker] = result
    return result


def simulate_fill(levels: list, count: int, limit_price: float) -> tuple:
    """
    Walk the order book levels to simulate a fill.

    Args:
        levels: list of (price, count) tuples, sorted ascending (best ask first)
        count: number of contracts to fill
        limit_price: maximum price willing to pay per contract

    Returns:
        (filled_count, avg_price) — avg_price is 0.0 if nothing filled
    """
    remaining = count
    total_cost = 0.0
    total_filled = 0

    for price, available in levels:
        if remaining <= 0:
            break
        if price > limit_price:
            break
        take = min(remaining, available)
        total_cost += take * price
        total_filled += take
        remaining -= take

    if total_filled == 0:
        return (0, 0.0)

    avg_price = total_cost / total_filled
    return (total_filled, round(avg_price, 6))


def best_ask(ticker: str, side: str = "yes") -> Optional[float]:
    """
    Return the cheapest available price from the orderbook for the given side.
    Returns None if the orderbook is empty or unavailable.
    """
    book = get_orderbook(ticker)
    if not book:
        return None
    key = "yes_levels" if side == "yes" else "no_levels"
    levels = book.get(key, [])
    if not levels:
        return None
    return levels[0][0]  # lowest price (best ask) is first after ascending sort
