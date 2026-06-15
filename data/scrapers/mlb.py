"""
data/scrapers/mlb.py — Full MLB historical data via pybaseball.
Statcast (2015-present, 8 seasons × ~10M rows/season),
FanGraphs batting/pitching (FIP, xFIP, wRC+, WHIP),
Baseball Reference game logs, park factors, umpire run environment.
"""
import time
import traceback
from typing import Optional

import pandas as pd

from config import MAX_RETRIES, BACKOFF_BASE

PARK_FACTORS = {
    "COL": 1.16, "BOS": 1.09, "CIN": 1.08, "TEX": 1.07, "MIL": 1.05,
    "PHI": 1.04, "HOU": 1.03, "NYY": 1.02, "ATL": 1.01, "CHC": 1.01,
    "LAD": 1.00, "NYM": 1.00, "STL": 0.99, "DET": 0.99, "ARI": 0.99,
    "MIN": 0.98, "CLE": 0.98, "SEA": 0.97, "SFG": 0.97, "TBR": 0.96,
    "BAL": 0.96, "MIA": 0.95, "OAK": 0.95, "SDP": 0.95, "LAA": 0.94,
    "PIT": 0.94, "TOR": 0.94, "KCR": 0.93, "WSN": 0.93, "CHW": 0.92,
}

HIGH_RUN_UMPS = frozenset({"bill miller", "dan bellino", "mike everitt", "joe west", "brian gorman"})
LOW_RUN_UMPS = frozenset({"angel hernandez", "rob drake", "mark carlson", "alan porter", "paul nauert"})

# pybaseball team abbreviation mapping → Statcast home_team codes
_SC_TEAM_MAP = {
    "ATL": "ATL", "ARI": "ARI", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC",
    "CWS": "CWS", "CHW": "CWS", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KC":  "KC",  "LAA": "LAA", "LAD": "LAD",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY",
    "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SD":  "SD",  "SEA": "SEA",
    "SFG": "SF",  "SF":  "SF",  "STL": "STL", "TB":  "TB",  "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSH", "WSN": "WSH",
}


def _sc_team(abbr: str) -> str:
    return _SC_TEAM_MAP.get(abbr, abbr)


# ── Statcast ───────────────────────────────────────────────────────────────────

def get_statcast_range(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Pitch-by-pitch Statcast for a date range."""
    try:
        from pybaseball import statcast
        df = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception as exc:
        print(f"Statcast error {start_dt}→{end_dt}: {exc}")
        return pd.DataFrame()


def get_statcast_season_chunked(year: int) -> pd.DataFrame:
    """Fetch full season Statcast in monthly chunks to avoid timeouts."""
    months = [
        (f"{year}-03-20", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-10-05"),
    ]
    frames = []
    for start, end in months:
        df = get_statcast_range(start, end)
        if not df.empty:
            frames.append(df)
            print(f"  [MLB] Statcast {start}→{end}: {len(df):,} pitches")
        time.sleep(1.5)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── FanGraphs advanced stats ───────────────────────────────────────────────────

def get_batting_stats_range(start_season: int, end_season: int,
                            qual: int = 1) -> pd.DataFrame:
    """FanGraphs advanced batting: wRC+, wOBA, BB%, K%, WAR."""
    try:
        from pybaseball import batting_stats
        df = batting_stats(start_season, end_season, qual=qual)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"pybaseball batting_stats error: {exc}")
        return pd.DataFrame()


def get_pitching_stats_range(start_season: int, end_season: int,
                             qual: int = 1) -> pd.DataFrame:
    """FanGraphs advanced pitching: FIP, xFIP, WHIP, K/9, BB/9, WAR."""
    try:
        from pybaseball import pitching_stats
        df = pitching_stats(start_season, end_season, qual=qual)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"pybaseball pitching_stats error: {exc}")
        return pd.DataFrame()


def get_team_batting_stats(start_season: int, end_season: int) -> pd.DataFrame:
    """FanGraphs team batting aggregates."""
    try:
        from pybaseball import team_batting
        df = team_batting(start_season, end_season)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"pybaseball team_batting error: {exc}")
        return pd.DataFrame()


def get_team_pitching_stats(start_season: int, end_season: int) -> pd.DataFrame:
    """FanGraphs team pitching aggregates."""
    try:
        from pybaseball import team_pitching
        df = team_pitching(start_season, end_season)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"pybaseball team_pitching error: {exc}")
        return pd.DataFrame()


def compute_fip(pitching_df: pd.DataFrame, pitcher_name: str) -> float:
    """Fielding Independent Pitching for a named pitcher from FanGraphs stats."""
    if pitching_df.empty or not pitcher_name:
        return 4.20
    last = pitcher_name.strip().split()[-1].lower()
    matches = pitching_df[pitching_df.get("Name", pd.Series()).str.lower().str.contains(last, na=False)]
    if matches.empty:
        return 4.20
    if "FIP" in matches.columns:
        return float(matches["FIP"].iloc[0])
    if "ERA" in matches.columns:
        return float(matches["ERA"].iloc[0])
    return 4.20


def compute_xfip(pitching_df: pd.DataFrame, pitcher_name: str) -> float:
    """xFIP for a named pitcher."""
    if pitching_df.empty or not pitcher_name:
        return 4.20
    last = pitcher_name.strip().split()[-1].lower()
    matches = pitching_df[pitching_df.get("Name", pd.Series()).str.lower().str.contains(last, na=False)]
    if matches.empty:
        return 4.20
    if "xFIP" in matches.columns:
        return float(matches["xFIP"].iloc[0])
    return compute_fip(pitching_df, pitcher_name)


# ── Schedule and record (Baseball Reference) ───────────────────────────────────

def get_schedule_and_record(team: str, season: int) -> pd.DataFrame:
    """Full game log for a team via pybaseball (Baseball Reference)."""
    try:
        from pybaseball import schedule_and_record
        df = schedule_and_record(season, team)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"pybaseball schedule_and_record {team} {season}: {exc}")
        return pd.DataFrame()


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_team_metrics(statcast_df: pd.DataFrame, team: str) -> dict:
    """Exit velocity and hard-hit rate for a team from Statcast."""
    if statcast_df.empty:
        return {"exit_velocity": 88.0, "hard_hit_rate": 0.35}
    sc_team = _sc_team(team)
    team_df = statcast_df[
        (statcast_df.get("home_team", pd.Series()) == sc_team) |
        (statcast_df.get("away_team", pd.Series()) == sc_team)
    ]
    batted = team_df[team_df["launch_speed"].notna()] if "launch_speed" in team_df.columns else pd.DataFrame()
    if batted.empty:
        return {"exit_velocity": 88.0, "hard_hit_rate": 0.35}
    return {
        "exit_velocity": float(batted["launch_speed"].mean()),
        "hard_hit_rate": float((batted["launch_speed"] >= 95).mean()),
    }


def compute_pitcher_whiff_rate(statcast_df: pd.DataFrame, pitcher_name: str) -> float:
    """Whiff rate (swinging strikes / swings) for a named pitcher."""
    if statcast_df.empty or not pitcher_name or "player_name" not in statcast_df.columns:
        return 0.25
    last = pitcher_name.strip().split()[-1].lower()
    p_df = statcast_df[statcast_df["player_name"].str.lower().str.contains(last, na=False)]
    if p_df.empty:
        return 0.25
    if "description" not in p_df.columns:
        return 0.25
    swings = p_df[p_df["description"].isin([
        "swinging_strike", "foul", "hit_into_play", "swinging_strike_blocked", "foul_tip"
    ])]
    whiffs = p_df[p_df["description"].isin(["swinging_strike", "swinging_strike_blocked"])]
    return float(len(whiffs) / len(swings)) if len(swings) > 0 else 0.25


def compute_team_wrc_plus(batting_df: pd.DataFrame, team: str) -> float:
    """Team wRC+ (weighted runs created+) from FanGraphs batting stats."""
    if batting_df.empty or "Team" not in batting_df.columns:
        return 100.0
    t_df = batting_df[batting_df["Team"].str.upper() == team.upper()]
    if t_df.empty or "wRC+" not in t_df.columns:
        return 100.0
    return float(t_df["wRC+"].mean())


def compute_team_war(pitching_df: pd.DataFrame, team: str) -> float:
    """Total pitching WAR for a team from FanGraphs."""
    if pitching_df.empty or "Team" not in pitching_df.columns:
        return 5.0
    t_df = pitching_df[pitching_df["Team"].str.upper() == team.upper()]
    if t_df.empty or "WAR" not in t_df.columns:
        return 5.0
    return float(t_df["WAR"].sum())


def compute_matchup_win_rate(statcast_df: pd.DataFrame, home_team: str) -> float:
    """Historical home team win rate from Statcast game records."""
    if statcast_df.empty or "home_team" not in statcast_df.columns:
        return 0.5
    sc_team = _sc_team(home_team)
    home_games = statcast_df[statcast_df["home_team"] == sc_team]
    if home_games.empty:
        return 0.5
    by_game = home_games.groupby("game_pk").agg(
        home_score=("home_score", "max"),
        away_score=("away_score", "max"),
    )
    wins = (by_game["home_score"] > by_game["away_score"]).mean()
    return float(wins) if not pd.isna(wins) else 0.5


def compute_umpire_run_factor(umpire: str) -> float:
    """Historical umpire run-environment tendency."""
    if not umpire:
        return 1.0
    name = umpire.lower().strip()
    if name in HIGH_RUN_UMPS:
        return 1.08
    if name in LOW_RUN_UMPS:
        return 0.93
    return 1.0


def get_park_factor(team: str) -> float:
    """Static park factor (run-normalized)."""
    return PARK_FACTORS.get(team, 1.0)


# ── Game-level aggregation from Statcast ──────────────────────────────────────

def statcast_to_game_logs(df: pd.DataFrame) -> list:
    """Aggregate pitch-level Statcast into game-level feature dicts."""
    if df.empty or "game_pk" not in df.columns:
        return []
    needed = {"game_pk", "game_date", "home_team", "away_team",
              "home_score", "away_score"}
    available = needed.intersection(df.columns)
    if len(available) < 4:
        return []

    logs = []
    grouped = df.groupby("game_pk")
    for game_pk, gdf in grouped:
        try:
            row = gdf.iloc[0]
            home_score = int(gdf["home_score"].max() if "home_score" in gdf else 0)
            away_score = int(gdf["away_score"].max() if "away_score" in gdf else 0)
            game_date = str(row.get("game_date", ""))[:10]
            home_team = str(row.get("home_team", ""))
            away_team = str(row.get("away_team", ""))
            if not game_date or not home_team or not away_team:
                continue
            h_metrics = compute_team_metrics(gdf, home_team)
            a_metrics = compute_team_metrics(gdf, away_team)
            year = int(game_date[:4]) if game_date else 0
            logs.append({
                "game_id": f"sc_{game_pk}",
                "game_date": game_date,
                "season": year,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_score > away_score,
                "home_exit_velocity": h_metrics["exit_velocity"],
                "home_hard_hit_rate": h_metrics["hard_hit_rate"],
                "away_exit_velocity": a_metrics["exit_velocity"],
                "away_hard_hit_rate": a_metrics["hard_hit_rate"],
                "park_factor": get_park_factor(home_team),
                "source": "statcast",
            })
        except Exception:
            continue
    return logs


# ── Pitcher-batter matchup aggregation ────────────────────────────────────────

def compute_pitcher_batter_matchups(statcast_df: pd.DataFrame,
                                    min_ab: int = 10) -> list:
    """
    Aggregate all pitcher-batter matchups with win rate, AB, K, BB.
    Returns list of matchup dicts for Supabase upsert.
    """
    if statcast_df.empty:
        return []
    needed = {"player_name", "batter", "events", "description"}
    if not needed.issubset(statcast_df.columns):
        return []
    try:
        at_bats = statcast_df[statcast_df["events"].notna()].copy()
        at_bats["hit"] = at_bats["events"].isin([
            "single", "double", "triple", "home_run"
        ]).astype(int)
        at_bats["k"] = (at_bats["events"] == "strikeout").astype(int)
        at_bats["bb"] = (at_bats["events"] == "walk").astype(int)
        at_bats["hr"] = (at_bats["events"] == "home_run").astype(int)

        grouped = at_bats.groupby(["player_name", "batter"]).agg(
            ab=("events", "count"),
            hits=("hit", "sum"),
            k=("k", "sum"),
            bb=("bb", "sum"),
            hr=("hr", "sum"),
        ).reset_index()

        grouped = grouped[grouped["ab"] >= min_ab]
        grouped["matchup_win_rate"] = grouped["hits"] / grouped["ab"]

        records = []
        for _, row in grouped.iterrows():
            records.append({
                "id": f"{row['player_name']}_{row['batter']}".replace(" ", "_")[:200],
                "pitcher": str(row["player_name"]),
                "batter": str(row["batter"]),
                "ab": int(row["ab"]),
                "hits": int(row["hits"]),
                "k": int(row["k"]),
                "bb": int(row["bb"]),
                "hr": int(row["hr"]),
                "matchup_win_rate": round(float(row["matchup_win_rate"]), 4),
            })
        return records
    except Exception as exc:
        print(f"compute_pitcher_batter_matchups error: {exc}")
        return []


# ── Full history ingestion ─────────────────────────────────────────────────────

def ingest_full_history(seasons: list, incremental_from=None) -> list:
    """
    Master MLB ingestion: Statcast (8 seasons × ~10M rows) + FanGraphs.
    Expects 2-3 hours on Modal CPU.
    """
    print(f"[MLB] Ingesting {len(seasons)} seasons ({seasons[0]}-{seasons[-1]})...")

    # FanGraphs batting and pitching (one call for full range)
    start_s, end_s = seasons[0], seasons[-1]
    print(f"[MLB] FanGraphs batting stats {start_s}-{end_s}...")
    batting_df = get_batting_stats_range(start_s, end_s)
    print(f"[MLB] FanGraphs pitching stats {start_s}-{end_s}...")
    pitching_df = get_pitching_stats_range(start_s, end_s)
    print(f"[MLB] Team batting stats...")
    team_bat_df = get_team_batting_stats(start_s, end_s)
    print(f"[MLB] Team pitching stats...")
    team_pit_df = get_team_pitching_stats(start_s, end_s)
    print(f"[MLB] FanGraphs: {len(batting_df)} batter rows, {len(pitching_df)} pitcher rows")

    all_logs = []
    all_statcast_frames = []

    for year in seasons:
        print(f"[MLB] Statcast {year} (chunked by month)...")
        sc_df = get_statcast_season_chunked(year)
        print(f"[MLB] Statcast {year}: {len(sc_df):,} pitches")

        if incremental_from:
            if not sc_df.empty and "game_date" in sc_df.columns:
                sc_df = sc_df[sc_df["game_date"].astype(str) > str(incremental_from)]

        if not sc_df.empty:
            all_statcast_frames.append(sc_df)
            logs = statcast_to_game_logs(sc_df)
            print(f"[MLB] Statcast {year}: {len(logs)} games")

            # Enrich with FanGraphs team-level stats
            for g in logs:
                h_team = g.get("home_team", "")
                a_team = g.get("away_team", "")
                g["home_wrc_plus"] = compute_team_wrc_plus(team_bat_df, h_team)
                g["away_wrc_plus"] = compute_team_wrc_plus(team_bat_df, a_team)
                g["home_pitching_war"] = compute_team_war(team_pit_df, h_team)
                g["away_pitching_war"] = compute_team_war(team_pit_df, a_team)

            all_logs.extend(logs)

    # Compute pitcher-batter matchups from all Statcast data and store
    if all_statcast_frames:
        print("[MLB] Computing pitcher-batter matchups across all seasons...")
        full_sc = pd.concat(all_statcast_frames, ignore_index=True)
        matchups = compute_pitcher_batter_matchups(full_sc, min_ab=10)
        if matchups:
            from db.historical_store import upsert_pitcher_matchups
            upsert_pitcher_matchups(matchups)
            print(f"[MLB] Stored {len(matchups):,} pitcher-batter matchup records")

    print(f"[MLB] Ingestion complete: {len(all_logs)} enriched game logs")
    return all_logs
