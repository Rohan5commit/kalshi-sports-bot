"""
data/scrapers/polymarket_history.py — Pull historical Polymarket sports markets.
CLOB API + Gamma API — fully public, zero authentication required.
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

CLOB = "https://clob.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"

SPORTS_KEYWORDS = [
    "nba", "nfl", "mlb", "nhl", "ncaa", "football", "basketball",
    "baseball", "hockey", "soccer", "ufc", "mma", "tennis", "golf",
    "super bowl", "world series", "stanley cup", "playoffs", "championship",
]


def _get(base: str, path: str, params: dict = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(f"{base}{path}", params=params or {}, timeout=30)
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


def get_all_sports_markets() -> list:
    markets = []
    # Gamma API: pull closed sports markets
    try:
        page, limit = 0, 500
        while True:
            data = _get(GAMMA, "/markets", {
                "closed": "true", "limit": limit, "offset": page * limit,
            })
            batch = data if isinstance(data, list) else data.get("markets", [])
            if not batch:
                break
            sports_batch = [
                m for m in batch
                if any(kw in (m.get("question", "") + m.get("description", "")).lower()
                       for kw in SPORTS_KEYWORDS)
            ]
            markets.extend(sports_batch)
            if len(batch) < limit:
                break
            page += 1
            time.sleep(0.3)
        print(f"[Polymarket] Gamma API: {len(markets)} sports markets found")
    except Exception as exc:
        print(f"[Polymarket] Gamma API error: {exc}")

    # Also try CLOB markets endpoint
    try:
        clob_markets = []
        next_cursor = None
        while True:
            params = {"limit": 500}
            if next_cursor:
                params["next_cursor"] = next_cursor
            data = _get(CLOB, "/markets", params)
            batch = data.get("data", [])
            sports_batch = [
                m for m in batch
                if any(kw in (m.get("question", "") + m.get("description", "")).lower()
                       for kw in SPORTS_KEYWORDS)
                and not m.get("active", True)  # closed markets only
            ]
            clob_markets.extend(sports_batch)
            next_cursor = data.get("next_cursor")
            if not next_cursor or not batch:
                break
            time.sleep(0.3)
        print(f"[Polymarket] CLOB API: {len(clob_markets)} additional sports markets")
        # Merge, deduplicate by condition_id
        existing_ids = {m.get("condition_id") or m.get("id") for m in markets}
        for m in clob_markets:
            mid = m.get("condition_id") or m.get("id")
            if mid not in existing_ids:
                markets.append(m)
                existing_ids.add(mid)
    except Exception as exc:
        print(f"[Polymarket] CLOB markets error: {exc}")

    return markets


def get_market_price_history(market_id: str) -> list:
    try:
        data = _get(CLOB, "/prices-history", {
            "market_id": market_id,
            "interval": "1h",
            "fidelity": 60,
        })
        return data.get("history", [])
    except Exception as exc:
        print(f"[Polymarket] price_history {market_id[:20]}: {exc}")
        return []


def get_market_trades(market_id: str) -> list:
    trades, cursor = [], None
    while True:
        params = {"market_id": market_id, "limit": 500}
        if cursor:
            params["next_cursor"] = cursor
        try:
            data = _get(CLOB, "/trades", params)
        except Exception:
            break
        batch = data.get("data", [])
        trades.extend(batch)
        cursor = data.get("next_cursor")
        if not cursor or not batch:
            break
    return trades


def _parse_ts(ts) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(ts)[:19], fmt[:19]).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _norm_price(p) -> float:
    if p is None:
        return 0.5
    f = float(p)
    # Polymarket CLOB prices are in [0, 1]
    return min(max(f, 0.0), 1.0)


def compute_market_features(history: list, game_start_time: str) -> dict:
    game_start = _parse_ts(game_start_time)
    price_cutoff = (game_start - timedelta(minutes=30)) if game_start else None

    timed = []
    for h in history:
        ts = _parse_ts(h.get("t") or h.get("timestamp"))
        if ts:
            p = _norm_price(h.get("p") or h.get("price"))
            vol = float(h.get("v") or h.get("volume") or 0)
            timed.append((ts, p, vol))
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
    total_volume = sum(t[2] for t in timed)
    last_hour_volume = 0.0
    if game_start:
        one_hour_before = game_start - timedelta(hours=1)
        last_hour_volume = sum(t[2] for t in timed
                               if one_hour_before <= t[0] <= (price_cutoff or game_start))
    price_at_tipoff = timed[-1][1]
    if game_start:
        at_tipoff = [t for t in timed if t[0] <= game_start]
        if at_tipoff:
            price_at_tipoff = at_tipoff[-1][1]

    return {
        "open_price": round(open_price, 4),
        "close_price": round(close_price, 4),
        "price_movement": round(close_price - open_price, 4),
        "total_volume": round(total_volume, 2),
        "last_hour_volume": round(last_hour_volume, 2),
        "price_at_tipoff": round(price_at_tipoff, 4),
        "max_single_move": round(max(moves), 4) if moves else 0.0,
        "days_open": (timed[-1][0] - timed[0][0]).days if len(timed) > 1 else 0,
    }


def ingest_polymarket_history() -> dict:
    from data.game_matcher import match_market_to_game
    from db.historical_store import (upsert_polymarket_price_history, upsert_polymarket_trades,
                                     upsert_polymarket_game_matches)

    print("[Polymarket] Fetching all sports markets...")
    all_markets = get_all_sports_markets()
    print(f"[Polymarket] Total sports markets: {len(all_markets)}")

    price_buf, trade_buf, match_records = [], [], []
    total_prices, total_trades = 0, 0

    for i, market in enumerate(all_markets):
        market_id = (market.get("condition_id") or market.get("id") or
                     market.get("market_id", ""))
        if not market_id:
            continue
        title = (market.get("question") or market.get("title") or
                 market.get("description", ""))
        close_time = (market.get("end_date_iso") or market.get("end_date") or
                      market.get("closed_time", ""))

        match = match_market_to_game(title=title, close_time=close_time,
                                     market_id=market_id, source="polymarket")
        game_start_time = (match.get("game_start_time", close_time)
                           if match else close_time)

        history = get_market_price_history(market_id)
        feats = compute_market_features(history, game_start_time) if history else {}
        total_prices += len(history)

        for h in history:
            ts = h.get("t") or h.get("timestamp", "")
            p = _norm_price(h.get("p") or h.get("price"))
            vol = float(h.get("v") or h.get("volume") or 0)
            price_buf.append({
                "market_id": market_id,
                "timestamp": ts if isinstance(ts, str) else datetime.fromtimestamp(
                    float(ts), tz=timezone.utc).isoformat(),
                "yes_price": p,
                "no_price": round(1.0 - p, 4),
                "volume": vol,
                "game_id": market_id if match else None,
            })

        trades = get_market_trades(market_id)
        total_trades += len(trades)
        for t in trades:
            trade_id = t.get("id") or f"{market_id}_{t.get('timestamp', i)}"
            ts = t.get("timestamp") or ""
            price_buf_ts = (datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
                            if isinstance(ts, (int, float)) else ts)
            trade_buf.append({
                "id": str(trade_id),
                "market_id": market_id,
                "timestamp": price_buf_ts,
                "price": _norm_price(t.get("price")),
                "size": float(t.get("size") or 0),
                "side": t.get("side", ""),
            })

        if match:
            match_records.append({
                "market_id": market_id,
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
            upsert_polymarket_price_history(price_buf); price_buf = []
        if len(trade_buf) >= 5000:
            upsert_polymarket_trades(trade_buf); trade_buf = []
        if (i + 1) % 100 == 0:
            print(f"[Polymarket] {i+1}/{len(all_markets)} markets processed")
            time.sleep(0.3)

    if price_buf: upsert_polymarket_price_history(price_buf)
    if trade_buf: upsert_polymarket_trades(trade_buf)
    if match_records: upsert_polymarket_game_matches(match_records)

    print(f"[Polymarket] Done: {len(all_markets)} markets, {total_prices} price ticks, "
          f"{total_trades} trades, {len(match_records)} game matches")
    return {"markets": len(all_markets), "price_records": total_prices,
            "trades": total_trades, "matches": len(match_records)}
