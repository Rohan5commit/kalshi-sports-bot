"""
trading/kalshi_client.py — Kalshi REST API wrapper.
Market data endpoints are public (no auth).
Trading endpoints require KALSHI_API_KEY from Modal Secrets.
"""
import os
import time
import traceback
import requests
from typing import Optional

from config import KALSHI_DEMO_BASE, KALSHI_PROD_BASE, KALSHI_USE_DEMO, MAX_RETRIES, BACKOFF_BASE
from db.supabase_client import log_error

BASE_URL = KALSHI_DEMO_BASE if KALSHI_USE_DEMO else KALSHI_PROD_BASE


def _public_request(method: str, endpoint: str, **kwargs) -> Optional[dict]:
    """Public market data request — no auth required."""
    url = f"{BASE_URL}{endpoint}"
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.request(method, url, timeout=20, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** attempt)
    log_error(
        context=f"kalshi_client._public_request {method} {endpoint}",
        error_msg=str(last_exc),
        tb=traceback.format_exc(),
    )
    return None


def _make_rsa_headers(method: str, endpoint: str) -> dict:
    """RSA-PSS signed headers for Kalshi production API."""
    import base64 as _b64
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend

    key_id = os.environ.get("KALSHI_API_KEY_ID", "")
    pem = os.environ.get("KALSHI_PRIVATE_KEY", "")
    if not key_id or not pem:
        return {"Content-Type": "application/json"}
    if "\\n" in pem:
        pem = pem.replace("\\n", "\n")
    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + endpoint).encode("utf-8")
    try:
        private_key = serialization.load_pem_private_key(
            pem.encode(), password=None, backend=default_backend()
        )
    except Exception as exc:
        log_error(context="kalshi_client._make_rsa_headers", error_msg=str(exc), tb="")
        return {"Content-Type": "application/json"}
    sig = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": _b64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }


def _authed_request(method: str, endpoint: str, **kwargs) -> Optional[dict]:
    """Authenticated request for trading endpoints using RSA-PSS signing."""
    url = f"{BASE_URL}{endpoint}"
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            headers = _make_rsa_headers(method, endpoint)
            resp = requests.request(method, url, headers=headers, timeout=20, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** attempt)
    log_error(
        context=f"kalshi_client._authed_request {method} {endpoint}",
        error_msg=str(last_exc),
        tb=traceback.format_exc(),
    )
    return None


# ── Public market data ─────────────────────────────────────────────────────────

def get_markets(status: str = "open", limit: int = 1000) -> list:
    """Fetch open sports markets."""
    data = _public_request("GET", "/markets", params={"status": status, "category": "sports", "limit": limit})
    return (data or {}).get("markets", [])


def get_market(market_ticker: str) -> Optional[dict]:
    """Fetch a single market by ticker."""
    data = _public_request("GET", f"/markets/{market_ticker}")
    return (data or {}).get("market")


def get_market_orderbook(market_ticker: str) -> Optional[dict]:
    """Fetch orderbook for precise implied probability."""
    return _public_request("GET", f"/markets/{market_ticker}/orderbook")


def get_implied_probability(market: dict) -> float:
    """
    Implied probability from Kalshi yes_ask price.
    yes_ask is the price to buy Yes, in cents (0-100).
    """
    yes_ask = market.get("yes_ask") or market.get("yes_price") or 50
    return float(yes_ask) / 100.0


def search_sports_markets(home_team: str, away_team: str, sport: str) -> list:
    """Search open sports markets for a specific game matchup."""
    all_markets = get_markets(limit=1000)
    home_lower = home_team.lower()
    away_lower = away_team.lower()
    sport_lower = sport.lower()
    matches = []
    for market in all_markets:
        combined = ((market.get("title") or "") + " " + (market.get("subtitle") or "")).lower()
        if (home_lower in combined or away_lower in combined) and sport_lower in combined:
            matches.append(market)
    return matches


# ── Authenticated trading endpoints ───────────────────────────────────────────

def get_balance() -> float:
    """Return available balance in USD."""
    data = _authed_request("GET", "/portfolio/balance")
    return float((data or {}).get("balance", 0)) / 100.0


def get_open_positions() -> list:
    data = _authed_request("GET", "/portfolio/positions")
    return (data or {}).get("market_positions", [])


def place_order(
    market_ticker: str,
    side: str,
    count: int,
    price: int,
    order_type: str = "limit",
) -> Optional[dict]:
    """Place a limit order on Kalshi demo. price in cents (1-99)."""
    payload = {
        "ticker": market_ticker,
        "action": "buy",
        "side": side,
        "type": order_type,
        "count": count,
        "yes_price": price if side == "yes" else 100 - price,
    }
    result = _authed_request("POST", "/portfolio/orders", json=payload)
    if result is None:
        log_error(
            context=f"kalshi_client.place_order({market_ticker}, {side})",
            error_msg="Kalshi API returned None — order not placed",
            tb="",
        )
    return result


def usd_to_contracts(usd_amount: float, price_cents: int) -> int:
    """Convert USD bet size to number of Kalshi contracts."""
    if price_cents <= 0:
        return 0
    return max(1, int(usd_amount / (price_cents / 100.0)))


def get_order_status(order_id: str) -> Optional[dict]:
    return _authed_request("GET", f"/portfolio/orders/{order_id}")


def cancel_order(order_id: str) -> Optional[dict]:
    return _authed_request("DELETE", f"/portfolio/orders/{order_id}")
