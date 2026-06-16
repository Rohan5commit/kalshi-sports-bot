"""
data/scrapers/kalshi_history.py — Pull historical Kalshi sports markets.
Uses public demo API — zero authentication required.
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

BASE = "https://demo-api.kalshi.co/trade-api/v2"

SPORTS_SERIES_PREFIXES = [
    "NBA", "NFL", "MLB", "NHL", "NCAAFB", "NCAABB", "WNBA", "MLS",
    "EPL", "UFC", "BOXING", "TENNIS", "GOLF", "SOCCER",
]


def _get(path: str, params: dict = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE}{path}", params=params or {}, timeout=30)
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
    try:
        data = _get("/series", {"limit": 200})
        return data.get("series", [])
    except Exception as exc:
        print(f"[Kalshi] get_all_series error: {exc}")
        return []


def get_series_markets(series_ticker: str) -> list:
    markets, cursor, page = [], None, 0
    while True:
        params = {"limit": 200, "status": "settled"}
        if cursor:
            params["cursor"] = cursor
        try:
            data = _get(f"/series/{series_ticker}/markets", params)
        except Exception as exc:
            print(f"[Kalshi] {series_ticker} error: {exc}")
            break
        batch = data.get("markets", [])
        markets.extend(batch)
        cursor = data.get("cursor")
        page += 1
        if not cursor or not batch:
            break
        if page % 5 == 0:
            time.sleep(0.5)
    return markets


def get_market_price_history(ticker: str) -> list:
    try:
        data = _get(f"/markets/{ticker}/history")
        return data.get("history", [])
    except Exception as exc:
        print(f"[Kalshi] price_history {ticker}: {exc}")
        return []


def get_market_trades(ticker: str) -> list:
    trades, cursor = [], None
    while True:
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
        if not cursor or not batch:
            break
    return trades


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(str(ts)[:19], fmt[:19]).replace(tzinfo=timezone.utc)
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


def ingest_kalshi_history() -> dict:
    from data.game_matcher import match_market_to_game
    from db.historical_store import (upsert_kalshi_price_history, upsert_kalshi_trades,
                                     upsert_kalshi_game_matches)

    print("[Kalshi] Discovering sports series...")
    all_series = get_all_series()
    sports_series = [
        s["ticker"] for s in all_series
        if any(pfx in s.get("ticker", "").upper() for pfx in SPORTS_SERIES_PREFIXES)
    ] or SPORTS_SERIES_PREFIXES

    print(f"[Kalshi] {len(sports_series)} sports series: {sports_series[:12]}")

    all_markets = []
    for sticker in sports_series:
        try:
            mkts = get_series_markets(sticker)
            print(f"[Kalshi] {sticker}: {len(mkts)} markets")
            all_markets.extend(mkts)
            time.sleep(0.2)
        except Exception as exc:
            print(f"[Kalshi] {sticker} error: {exc}")

    print(f"[Kalshi] Total markets: {len(all_markets)}")
    price_buf, trade_buf, match_records = [], [], []
    total_prices, total_trades = 0, 0

    for i, market in enumerate(all_markets):
        ticker = market.get("ticker", "")
        if not ticker:
            continue
        title = market.get("title") or market.get("subtitle", "")
        close_time = market.get("close_time") or market.get("expiration_time", "")

        match = match_market_to_game(title=title, close_time=close_time,
                                     market_id=ticker, source="kalshi")
        game_start_time = (match.get("game_start_time", close_time)
                           if match else close_time)

        history = get_market_price_history(ticker)
        feats = compute_market_features(history, game_start_time) if history else {}
        total_prices += len(history)

        for h in history:
            ts = h.get("ts") or h.get("timestamp", "")
            price_buf.append({
                "ticker": ticker,
                "timestamp": ts,
                "yes_price": _norm_price(h.get("yes_price")),
                "no_price": _norm_price(h.get("no_price")),
                "volume": int(h.get("volume") or 0),
                "game_id": ticker if match else None,
            })

        trades = get_market_trades(ticker)
        total_trades += len(trades)
        for t in trades:
            trade_id = t.get("trade_id") or f"{ticker}_{t.get('created_time', i)}"
            trade_buf.append({
                "id": trade_id,
                "ticker": ticker,
                "timestamp": t.get("created_time", ""),
                "price": _norm_price(t.get("yes_price")),
                "count": int(t.get("count") or 0),
                "taker_side": t.get("taker_side", ""),
            })

        if match:
            match_records.append({
                "kalshi_ticker": ticker,
                "espn_game_id": match.get("espn_game_id", ""),
                "sport": match.get("sport", ""),
                "game_date": match.get("game_date", ""),
                "home_team": match.get("home_team", ""),
                "away_team": match.get("away_team", ""),
                "game_start_time": game_start_time,
                "match_confidence_score": match.get("match_confidence_score", 0.0),
                **feats,
            })

        if len(price_buf) >= 5000:
            upsert_kalshi_price_history(price_buf); price_buf = []
        if len(trade_buf) >= 5000:
            upsert_kalshi_trades(trade_buf); trade_buf = []
        if (i + 1) % 100 == 0:
            print(f"[Kalshi] {i+1}/{len(all_markets)} markets processed")
            time.sleep(0.3)

    if price_buf: upsert_kalshi_price_history(price_buf)
    if trade_buf: upsert_kalshi_trades(trade_buf)
    if match_records: upsert_kalshi_game_matches(match_records)

    print(f"[Kalshi] Done: {len(all_markets)} markets, {total_prices} price ticks, "
          f"{total_trades} trades, {len(match_records)} game matches")
    return {"markets": len(all_markets), "price_records": total_prices,
            "trades": total_trades, "matches": len(match_records)}
