"""
trading/kalshi_client.py — Kalshi REST API wrapper.
Uses demo account by default; all credentials come from Modal Secrets.
Implements retry logic with exponential backoff.
"""
import os
import time
import traceback
import requests
from typing import Optional

from config import KALSHI_DEMO_BASE, KALSHI_PROD_BASE, KALSHI_USE_DEMO, MAX_RETRIES, BACKOFF_BASE
from db.supabase_client import log_error

BASE_URL = KALSHI_DEMO_BASE if KALSHI_USE_DEMO else KALSHI_PROD_BASE


def _get_headers() -> dict:
    api_key = os.environ["KALSHI_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _request(method: str, endpoint: str, **kwargs) -> Optional[dict]:
    """Generic HTTP request with retry logic."""
    url = f"{BASE_URL}{endpoint}"
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.request(
                method,
                url,
                headers=_get_headers(),
                timeout=20,
                **kwargs,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** attempt)
    log_error(
        context=f"kalshi_client._request {method} {endpoint}",
        error_msg=str(last_exc),
        tb=traceback.format_exc(),
    )
    return None


# ── Market data ────────────────────────────────────────────────────────────────

def get_markets(sport_tag: str = None, status: str = "open", limit: int = 100) -> list:
    """
    Fetch open markets, optionally filtered by sport tag.
    Returns list of market dicts.
    """
    params = {"status": status, "limit": limit}
    if sport_tag:
        params["series_ticker"] = sport_tag
    data = _request("GET", "/markets", params=params)
    if data is None:
        return []
    return data.get("markets", [])


def get_market(market_ticker: str) -> Optional[dict]:
    """Fetch a single market by ticker."""
    return _request("GET", f"/markets/{market_ticker}")


def get_market_orderbook(market_ticker: str) -> Optional[dict]:
    """Fetch order book for a market."""
    return _request("GET", f"/markets/{market_ticker}/orderbook")


def search_sports_markets(home_team: str, away_team: str, sport: str) -> list:
    """
    Search for Kalshi markets matching a specific game matchup.
    Looks for markets containing team names in the title.
    """
    all_markets = get_markets(limit=200)
    if not all_markets:
        return []

    sport_lower = sport.lower()
    home_lower = home_team.lower()
    away_lower = away_team.lower()

    matches = []
    for market in all_markets:
        title = (market.get("title") or "").lower()
        subtitle = (market.get("subtitle") or "").lower()
        combined = title + " " + subtitle
        if (home_lower in combined or away_lower in combined) and sport_lower in combined:
            matches.append(market)
    return matches


def get_implied_probability(market: dict) -> float:
    """
    Extract the implied probability from a Kalshi market's yes_price.
    yes_price is in cents (0–100).
    """
    yes_price = market.get("yes_price") or market.get("yes_ask") or 50
    return float(yes_price) / 100.0


# ── Account and balance ────────────────────────────────────────────────────────

def get_balance() -> float:
    """Return available balance in USD."""
    data = _request("GET", "/portfolio/balance")
    if not data:
        return 0.0
    # Balance returned in cents by Kalshi
    return float(data.get("balance", 0)) / 100.0


def get_open_positions() -> list:
    """Fetch open positions from the portfolio."""
    data = _request("GET", "/portfolio/positions")
    if not data:
        return []
    return data.get("market_positions", [])


# ── Order placement ────────────────────────────────────────────────────────────

def place_order(
    market_ticker: str,
    side: str,          # "yes" or "no"
    count: int,         # number of contracts
    price: int,         # in cents (1–99)
    order_type: str = "limit",
) -> Optional[dict]:
    """
    Place a limit order on Kalshi demo.
    Returns the order response dict, or None if the API is unreachable.
    """
    payload = {
        "ticker": market_ticker,
        "action": "buy",
        "side": side,
        "type": order_type,
        "count": count,
        "yes_price": price if side == "yes" else 100 - price,
    }
    result = _request("POST", "/portfolio/orders", json=payload)
    if result is None:
        log_error(
            context=f"kalshi_client.place_order({market_ticker}, {side})",
            error_msg="Kalshi API returned None — order not placed",
            tb="",
        )
    return result


def usd_to_contracts(usd_amount: float, price_cents: int) -> int:
    """
    Convert a USD bet size to number of Kalshi contracts.
    Each contract costs price_cents / 100 USD.
    """
    if price_cents <= 0:
        return 0
    cost_per_contract = price_cents / 100.0
    return max(1, int(usd_amount / cost_per_contract))


def get_order_status(order_id: str) -> Optional[dict]:
    """Fetch status of a specific order."""
    return _request("GET", f"/portfolio/orders/{order_id}")


def cancel_order(order_id: str) -> Optional[dict]:
    """Cancel an open order."""
    return _request("DELETE", f"/portfolio/orders/{order_id}")
