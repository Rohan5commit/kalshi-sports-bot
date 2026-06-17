"""
data/scrapers/kalshi_history.py — Pull historical Kalshi sports markets.
Production API with RSA authentication (KALSHI-ACCESS-KEY / KALSHI-ACCESS-SIGNATURE).
Fetches series, markets, price history, and trades for NBA/NFL/MLB games.
"""
import base64
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

API_HOST = "https://trading-api.kalshi.com"
API_PREFIX = "/trade-api/v2"
PROD_BASE = API_HOST + API_PREFIX

SPORTS_SERIES_PREFIXES = [
    "NBA", "NFL", "MLB", "NHL", "NCAA", "WNBA", "MLS",
    "EPL", "UFC", "BOXING", "TENNIS", "GOLF", "SOCCER",
]

_TARGET_SPORT_RE = re.compile(r"^KX(NBA|NFL|MLB)", re.I)
_PROP_SUFFIX_RE = re.compile(r"LEADER|MENTION|HIT$|3D$|3PT$", re.I)


def _make_auth_headers(method: str, path: str) -> dict:
    """RSA-sign a Kalshi production API request."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend

    key_id = os.environ.get("KALSHI_API_KEY_ID", "")
    pem = os.environ.get("KALSHI_PRIVATE_KEY", "")
    if not key_id or not pem:
        return {"Content-Type": "application/json"}

    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + API_PREFIX + path).encode("utf-8")
    private_key = serialization.load_pem_private_key(
        pem.encode(), password=None, backend=default_backend()
    )
    sig = private_key.sign(msg, padding.PKCS1v15(), hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }


def _get(path: str, params: dict = None, retries: int = 3) -> dict:
    params = params or {}
    for attempt in range(retries):
        try:
            headers = _make_auth_headers("GET", path)
            r = requests.get(f"{PROD_BASE}{path}", params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def get_all_series() -> list:
    """Get all sports-relevant series from the /series endpoint."""
    try:
        data = _get("/series", {"limit": 200})
        all_series = data.get("series", [])
        sports = [
            s for s in all_series
            if any(pfx in s.get("ticker", "").upper() for pfx in SPORTS_SERIES_PREFIXES)
        ]
        return sports
    except Exception as exc:
        print(f"[Kalshi] get_all_series error: {exc}")
        return []


def get_markets_for_series(series_ticker: str) -> list:
    """Get all settled markets for a series using GET /markets?series_ticker=."""
    markets, cursor = [], None
    while True:
        params = {"limit": 200, "series_ticker": series_ticker, "status": "settled"}
        if cursor:
            params["cursor"] = cursor
        try:
            data = _get("/markets", params)
        except Exception as exc:
            print(f"[Kalshi] {series_ticker} markets error: {exc}")
            break
        batch = data.get("markets", [])
        if not batch and not cursor:
            try:
                data2 = _get("/markets", {"limit": 200, "series_ticker": series_ticker})
                batch = data2.get("markets", [])
                markets.extend(batch)
            except Exception:
                pass
            break
        markets.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
    return markets


def get_all_sports_markets_paginated() -> list:
    """Fall-through: GET /markets with pagination, filter by title for sports."""
    markets, cursor, page = [], None, 0
    sports_kw = ["nba", "nfl", "mlb", "nhl", "ncaa", "mls", "ufc", "playoffs",
                 "championship", "super bowl", "world series"]
    while True:
        params = {"limit": 200, "status": "settled"}
        if cursor:
            params["cursor"] = cursor
        try:
            data = _get("/markets", params)
        except Exception as exc:
            print(f"[Kalshi] paginated markets error: {exc}")
            break
        batch = data.get("markets", [])
        for m in batch:
            title = (m.get("title") or m.get("subtitle") or "").lower()
            ticker = m.get("ticker", "").upper()
            if (any(kw in title for kw in sports_kw) or
                    any(pfx in ticker for pfx in SPORTS_SERIES_PREFIXES)):
                markets.append(m)
        cursor = data.get("cursor")
        page += 1
        if not cursor or not batch:
            break
        if page % 10 == 0:
            print(f"[Kalshi] paginated: {page} pages, {len(markets)} sports markets so far")
            time.sleep(0.5)
    return markets


def get_market_price_history(ticker: str) -> list:
    try:
        data = _get(f"/markets/{ticker}/history", {"limit": 1000})
        return data.get("history", [])
    except Exception as exc:
        print(f"[Kalshi] price_history {ticker}: {exc}")
        return []


def get_market_trades(ticker: str, max_pages: int = 20) -> list:
    trades, cursor, page = [], None, 0
    while page < max_pages:
        params = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor
        try:
            data = _get(f"/markets/{ticker}/trades", params)
        except Exception:
            break
        batch = data.get("trades", [])
        trades.extend(batch)
        cursor = data.get("cursor")
        page += 1
        if not cursor or not batch:
            break
    return trades


def _parse_ts(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    ts_str = str(ts)[:19]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _norm_price(p) -> float:
    if p is None:
        return 0.5
    f = float(p)
    return f / 100.0 if f > 1.0 else f


def compute_market_features(history: list, game_start_time: str) -> dict:
    game_start = _parse_ts(game_start_time)
    price_cutoff = (game_start - timedelta(minutes=30)) if game_start else None

    timed = []
    for h in history:
        ts = _parse_ts(h.get("ts") or h.get("timestamp"))
        if ts:
            timed.append((ts,
                          _norm_price(h.get("yes_price")),
                          _norm_price(h.get("no_price")),
                          int(h.get("volume") or 0)))
    timed.sort(key=lambda x: x[0])
    if not timed:
        return {}

    open_price = timed[0][1]
    close_price = open_price
    if price_cutoff:
        before = [t for t in timed if t[0] <= price_cutoff]
        if before:
            close_price = before[-1][1]

    moves = [abs(timed[i][1] - timed[i-1][1]) for i in range(1, len(timed))]
    total_volume = sum(t[3] for t in timed)
    last_hour_volume = 0
    if game_start:
        one_hour_before = game_start - timedelta(hours=1)
        last_hour_volume = sum(t[3] for t in timed
                               if one_hour_before <= t[0] <= (price_cutoff or game_start))
    price_at_tipoff = timed[-1][1]
    if game_start:
        at_tipoff = [t for t in timed if t[0] <= game_start]
        if at_tipoff:
            price_at_tipoff = at_tipoff[-1][1]
    last_no = timed[-1][2]
    overround = (timed[-1][1] + last_no - 1.0) if last_no > 0 else 0.0

    return {
        "open_price": round(open_price, 4),
        "close_price": round(close_price, 4),
        "price_movement": round(close_price - open_price, 4),
        "total_volume": float(total_volume),
        "last_hour_volume": float(last_hour_volume),
        "price_at_tipoff": round(price_at_tipoff, 4),
        "max_single_move": round(max(moves), 4) if moves else 0.0,
        "days_open": (timed[-1][0] - timed[0][0]).days if len(timed) > 1 else 0,
        "overround": round(overround, 4),
    }


def _process_single_market(market: dict):
    """Thread worker: pre-filter → match → price history + trades → compute features."""
    from data.game_matcher import match_market_to_game

    ticker = market.get("ticker", "")
    if not ticker:
        return None
    title = market.get("title") or market.get("subtitle", "")
    close_time = market.get("close_time") or market.get("expiration_time", "")

    title_lower = title.lower()
    if " vs" not in title_lower and " @ " not in title_lower and " at " not in title_lower:
        return None

    match = match_market_to_game(title=title, close_time=close_time,
                                  market_id=ticker, source="kalshi")
    if not match:
        return None

    game_start_time = match.get("game_start_time", close_time)

    # Fetch price history and trades from production API
    history = get_market_price_history(ticker)
    feats = compute_market_features(history, game_start_time) if history else {}

    price_recs = []
    for h in history:
        ts_val = h.get("ts") or h.get("timestamp")
        if ts_val is None:
            continue
        ts_dt = _parse_ts(ts_val)
        yes_p = _norm_price(h.get("yes_price"))
        no_p = _norm_price(h.get("no_price"))
        price_recs.append({
            "id": str(uuid.uuid4()),
            "ticker": ticker,
            "timestamp": ts_dt.isoformat() if ts_dt else None,
            "yes_price": yes_p,
            "no_price": no_p,
            "volume": int(h.get("volume") or 0),
            "game_id": match.get("espn_game_id", ""),
        })

    trades_raw = get_market_trades(ticker)
    trade_recs = []
    for t in trades_raw:
        trade_id = t.get("trade_id") or t.get("id", "")
        if not trade_id:
            continue
        ts_val = t.get("created_time") or t.get("timestamp", "")
        ts_dt = _parse_ts(ts_val)
        yes_p = t.get("yes_price", 50)
        trade_recs.append({
            "id": str(trade_id),
            "ticker": ticker,
            "timestamp": ts_dt.isoformat() if ts_dt else None,
            "price": _norm_price(yes_p),
            "count": int(t.get("count") or 0),
            "taker_side": t.get("taker_side", ""),
        })

    match_rec = {
        "kalshi_ticker": ticker,
        "espn_game_id": match.get("espn_game_id", ""),
        "sport": match.get("sport", ""),
        "game_date": match.get("game_date", ""),
        "home_team": match.get("home_team", ""),
        "away_team": match.get("away_team", ""),
        "game_start_time": game_start_time,
        "match_confidence_score": match.get("match_confidence_score", 0.0),
        **feats,
    }
    return (match_rec, price_recs, trade_recs)


def ingest_kalshi_history() -> dict:
    from db.historical_store import (upsert_kalshi_price_history, upsert_kalshi_trades,
                                     upsert_kalshi_game_matches)

    print("[Kalshi] Fetching sports series...")
    all_series = get_all_series()

    game_series = [
        s for s in all_series
        if _TARGET_SPORT_RE.match(s.get("ticker", ""))
        and not _PROP_SUFFIX_RE.search(s.get("ticker", ""))
    ]
    print(f"[Kalshi] {len(all_series)} sports series -> {len(game_series)} after NBA/NFL/MLB + prop filter")

    all_markets = []
    completed = 0

    def _fetch(ticker):
        return ticker, get_markets_for_series(ticker)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch, s.get("ticker", "")): s for s in game_series}
        for fut in as_completed(futures):
            ticker, mkts = fut.result()
            completed += 1
            if mkts:
                print(f"[Kalshi] {ticker}: {len(mkts)} markets")
                all_markets.extend(mkts)
            if completed % 50 == 0:
                print(f"[Kalshi] {completed}/{len(game_series)} series fetched...")

    if not all_markets:
        print("[Kalshi] Series approach yielded 0 markets, trying paginated scan...")
        all_markets = get_all_sports_markets_paginated()

    print(f"[Kalshi] Total game-level markets to process: {len(all_markets)}")
    if not all_markets:
        print("[Kalshi] No markets found")
        return {"markets": 0, "price_records": 0, "trades": 0, "matches": 0}

    from db.historical_store import ensure_espn_cache
    ensure_espn_cache()

    price_buf, trade_buf, match_records = [], [], []
    total_prices, total_trades, total_matches, processed = 0, 0, 0, 0

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_process_single_market, m): m for m in all_markets}
        for fut in as_completed(futures):
            processed += 1
            try:
                result = fut.result()
            except Exception as exc:
                print(f"[Kalshi] market error: {exc}")
                continue

            if result:
                match_rec, price_recs, trade_recs = result
                match_records.append(match_rec)
                price_buf.extend(price_recs)
                trade_buf.extend(trade_recs)
                total_prices += len(price_recs)
                total_trades += len(trade_recs)

            if len(price_buf) >= 5000:
                upsert_kalshi_price_history(price_buf); price_buf = []
            if len(trade_buf) >= 5000:
                upsert_kalshi_trades(trade_buf); trade_buf = []
            if len(match_records) >= 200:
                upsert_kalshi_game_matches(match_records)
                total_matches += len(match_records)
                print(f"[Kalshi] Flushed {total_matches} matches to DB ({processed}/{len(all_markets)} processed)")
                match_records = []

            if processed % 500 == 0:
                print(f"[Kalshi] {processed}/{len(all_markets)} markets processed, "
                      f"{total_matches + len(match_records)} matches, {total_prices} price ticks")

    if price_buf: upsert_kalshi_price_history(price_buf)
    if trade_buf: upsert_kalshi_trades(trade_buf)
    if match_records:
        upsert_kalshi_game_matches(match_records)
        total_matches += len(match_records)

    print(f"[Kalshi] Done: {len(all_markets)} markets scanned, {total_prices} price ticks, "
          f"{total_trades} trades, {total_matches} game matches")
    return {"markets": len(all_markets), "price_records": total_prices,
            "trades": total_trades, "matches": total_matches}
