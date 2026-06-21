"""
trading/kalshi_client.py — Kalshi REST API wrapper.
Market data endpoints are public (no auth).
Trading endpoints require KALSHI_API_KEY from Modal Secrets.
"""
import os
import re
import time
import traceback
import requests
from typing import Optional

from config import KALSHI_DEMO_BASE, KALSHI_PROD_BASE, KALSHI_USE_DEMO, MAX_RETRIES, BACKOFF_BASE
from db.supabase_client import log_error

BASE_URL = KALSHI_DEMO_BASE if KALSHI_USE_DEMO else KALSHI_PROD_BASE

# Kalshi event series tickers for each sport's individual game markets
_GAME_SERIES: dict = {
    "mlb": "KXMLBGAME",
    "nba": "KXNBAGAME",
    "nfl": "KXNFLGAME",
}

# Fallback keyword terms for older-style individual markets (title-based search)
_SPORT_KALSHI_TERMS: dict = {
    "mlb": ["baseball", "mlb"],
    "nba": ["basketball", "nba"],
    "nfl": ["american football", "football", "nfl"],
}


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
    """Fetch open sports markets (bundle/cross-category format)."""
    data = _public_request("GET", "/markets", params={"status": status, "category": "sports", "limit": limit})
    return (data or {}).get("markets", [])


def get_market(market_ticker: str) -> Optional[dict]:
    """Fetch a single market by ticker."""
    data = _public_request("GET", f"/markets/{market_ticker}")
    return (data or {}).get("market")


def get_market_orderbook(market_ticker: str) -> Optional[dict]:
    """Fetch orderbook for precise implied probability."""
    return _public_request("GET", f"/markets/{market_ticker}/orderbook")


def get_yes_price_cents(market: dict) -> int:
    """Return the yes ask price in integer cents (1-99), handling all known field formats."""
    # Current format: yes_ask_dollars (string USD, e.g. "0.5600")
    for key in ("yes_ask_dollars", "last_price_dollars"):
        val = market.get(key)
        if val:
            try:
                cents = round(float(val) * 100)
                if 1 <= cents <= 99:
                    return cents
            except (ValueError, TypeError):
                pass
    # Legacy format: yes_ask or yes_price (integer cents)
    for key in ("yes_ask", "yes_price"):
        val = market.get(key)
        if val:
            try:
                cents = int(val)
                if 1 <= cents <= 99:
                    return cents
            except (ValueError, TypeError):
                pass
    return 50


def get_implied_probability(market: dict) -> float:
    """Implied probability from the yes ask price."""
    return get_yes_price_cents(market) / 100.0


def parse_bundle_legs(title: str) -> list:
    """Parse a KXMVESPORTSMULTIGAMEEXTENDED title into individual leg dicts."""
    legs = []
    for part in (title or "").split(","):
        part = part.strip()
        if part.startswith("yes "):
            legs.append({"direction": "yes", "description": part[4:].strip()})
        elif part.startswith("no "):
            legs.append({"direction": "no", "description": part[3:].strip()})
    return legs


def _city_tokens(team_name: str) -> list:
    """Multi-word city prefix of a team name for fuzzy matching."""
    words = team_name.lower().split()
    tokens = []
    if len(words) >= 2:
        tokens.append(f"{words[0]} {words[1]}")
    if words and len(words[0]) >= 5:
        tokens.append(words[0])
    return tokens


def _get_game_events(sport: str) -> list:
    """Fetch all open game events for a sport from Kalshi's dedicated game series."""
    series = _GAME_SERIES.get(sport.lower())
    if not series:
        return []
    data = _public_request("GET", "/events", params={
        "status": "open", "series_ticker": series, "limit": 100,
    })
    return (data or {}).get("events", [])


def _get_event_markets(event_ticker: str) -> list:
    """Fetch all markets within a specific game event."""
    data = _public_request("GET", "/markets", params={
        "event_ticker": event_ticker, "limit": 10,
    })
    return (data or {}).get("markets", [])


def _home_market_from_event(event_ticker: str, markets: list) -> Optional[dict]:
    """
    Return the market where yes = home team wins.
    Convention: event ticker ends with {AWAY_CODE}{HOME_CODE}.
    The home market's ticker suffix equals HOME_CODE (the trailing part of teams_code).
    e.g. KXMLBGAME-26JUN232145ATHSF → teams_code=ATHSF → home suffix=SF.
    """
    # Extract the teams_code by stripping the date+time prefix from the last segment
    seg = event_ticker.split("-")[-1]
    teams_code = re.sub(r'^\d{2}\w{3}\d{2}(?:\d{4})?', '', seg)
    if not teams_code:
        return markets[0] if markets else None
    for m in markets:
        suffix = (m.get("ticker") or "").split("-")[-1]
        if suffix and teams_code.endswith(suffix):
            return m
    return markets[0] if markets else None


def _event_matches_game(event: dict, home_tokens: list, away_tokens: list,
                         home_lower: str, away_lower: str) -> bool:
    """
    Check whether a Kalshi event corresponds to our home vs away game.
    Event title format: "Away vs Home".
    """
    title = (event.get("title") or "").lower()
    sides = [s.strip() for s in title.split(" vs ")]
    if len(sides) < 2:
        return False
    away_side, home_side = sides[0], sides[-1]

    home_match = (
        any(t in home_side for t in home_tokens) or
        home_lower in home_side or
        any(w in home_lower for w in home_side.split() if len(w) >= 4)
    )
    away_match = (
        any(t in away_side for t in away_tokens) or
        away_lower in away_side or
        any(w in away_lower for w in away_side.split() if len(w) >= 4)
    )
    return home_match or away_match


def search_sports_markets(home_team: str, away_team: str, sport: str) -> list:
    """
    Find the Kalshi market(s) for a specific game.

    Search order:
    1. Dedicated game event series (KXMLBGAME / KXNBAGAME / KXNFLGAME) — returns the
       home-team-wins market so final_prob (P(home wins)) maps directly to yes_price.
    2. Individual title-based markets (older Kalshi format, unlikely but kept as fallback).
    3. Multi-game bundle/parlay markets (current Kalshi default for non-game content).
    """
    home_lower = home_team.lower()
    away_lower = away_team.lower()
    home_tokens = _city_tokens(home_team)
    away_tokens = _city_tokens(away_team)

    # ── Priority 1: dedicated game event series ────────────────────────────────
    events = _get_game_events(sport)
    for event in events:
        if _event_matches_game(event, home_tokens, away_tokens, home_lower, away_lower):
            event_ticker = event.get("event_ticker", "")
            markets = _get_event_markets(event_ticker)
            home_market = _home_market_from_event(event_ticker, markets)
            if home_market:
                print(f"[kalshi] matched game event {event_ticker} → {home_market.get('ticker')}")
                return [home_market]

    # ── Priority 2: individual title-based markets ─────────────────────────────
    all_markets = get_markets(limit=1000)
    sport_terms = _SPORT_KALSHI_TERMS.get(sport.lower(), [sport.lower()])
    individual = []
    for market in all_markets:
        combined = ((market.get("title") or "") + " " + (market.get("subtitle") or "")).lower()
        if (home_lower in combined or away_lower in combined) and any(t in combined for t in sport_terms):
            individual.append(market)
    if individual:
        return individual

    # ── Priority 3: bundle/parlay markets ─────────────────────────────────────
    bundle_matches = []
    for market in all_markets:
        ticker = market.get("ticker", "")
        if "MULTIGAME" not in ticker and "CROSSCATEGORY" not in ticker:
            continue
        title = (market.get("title") or "").lower()
        for leg in parse_bundle_legs(title):
            desc = leg["description"]
            if any(t in desc for t in home_tokens) or any(t in desc for t in away_tokens):
                market["_is_bundle"] = True
                bundle_matches.append(market)
                break
    return bundle_matches


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
    """Place a limit order on Kalshi. price in cents (1-99)."""
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
