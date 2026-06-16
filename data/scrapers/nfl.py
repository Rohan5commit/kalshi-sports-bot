"""
data/scrapers/nfl.py — Full NFL historical data via nfl_data_py.
Pulls play-by-play (300+ cols), participation, NextGen stats,
rosters, injuries, depth charts, contracts. EPA as DVOA proxy.
"""
import traceback
from typing import Optional

import pandas as pd

from config import MAX_RETRIES


def _safe_import(seasons: list, fn_name: str, **kwargs) -> pd.DataFrame:
    """Call any nfl_data_py import function with error handling."""
    import nfl_data_py as nfl
    fn = getattr(nfl, fn_name)
    try:
        return fn(seasons, **kwargs) if seasons else fn(**kwargs)
    except Exception as exc:
        print(f"nfl_data_py {fn_name} error: {exc}")
        return pd.DataFrame()


# ── Core data pulls ────────────────────────────────────────────────────────────

def get_schedules(seasons: list) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        return nfl.import_schedules(seasons)
    except Exception as exc:
        print(f"nfl_data_py schedules error: {exc}")
        return pd.DataFrame()


def get_pbp(seasons: list) -> pd.DataFrame:
    """Full play-by-play with all 300+ columns — large but fast (parquet downloads)."""
    import nfl_data_py as nfl
    try:
        return nfl.import_pbp_data(seasons, downcast=True)
    except Exception as exc:
        print(f"nfl_data_py pbp error: {exc}")
        return pd.DataFrame()


def get_pbp_slim(seasons: list) -> pd.DataFrame:
    """Slimmed PBP with only EPA-relevant columns for quick feature computation."""
    import nfl_data_py as nfl
    cols = [
        "game_id", "posteam", "defteam", "epa", "season", "week",
        "home_team", "away_team", "game_date", "down", "wp", "def_wp",
        "pass_attempt", "rush_attempt", "sack", "interception", "fumble_lost",
        "touchdown", "score_differential", "half_seconds_remaining",
        "play_type", "yards_gained",
    ]
    try:
        return nfl.import_pbp_data(seasons, columns=cols, downcast=True)
    except Exception as exc:
        print(f"nfl_data_py pbp_slim error: {exc}")
        return pd.DataFrame()


def get_participation(seasons: list) -> pd.DataFrame:
    """
    Every player on every play (~50M rows for full history).
    Used to compute player participation rates on high-leverage plays.
    """
    import nfl_data_py as nfl
    try:
        return nfl.import_participation(seasons, include_pbp=False)
    except Exception as exc:
        print(f"nfl_data_py participation error: {exc}")
        return pd.DataFrame()


def get_nextgen_stats(seasons: list, stat_type: str = "passing") -> pd.DataFrame:
    """NextGen Stats: passing (time_to_throw), rushing (efficiency), receiving (separation)."""
    import nfl_data_py as nfl
    try:
        return nfl.import_nextgen_stats(seasons, stat_type=stat_type)
    except Exception as exc:
        print(f"nfl_data_py nextgen_{stat_type} error: {exc}")
        return pd.DataFrame()


def get_rosters(seasons: list) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        return nfl.import_rosters(seasons)
    except Exception as exc:
        print(f"nfl_data_py rosters error: {exc}")
        return pd.DataFrame()


def get_injuries(seasons: list) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        return nfl.import_injuries(seasons)
    except Exception as exc:
        return pd.DataFrame()


def get_depth_charts(seasons: list) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        return nfl.import_depth_charts(seasons)
    except Exception as exc:
        return pd.DataFrame()


def get_contracts() -> pd.DataFrame:
    """Salary cap data per player from OverTheCap (no season argument)."""
    import nfl_data_py as nfl
    try:
        return nfl.import_contracts()
    except Exception as exc:
        print(f"nfl_data_py contracts error: {exc}")
        return pd.DataFrame()


def get_weekly_stats(seasons: list) -> pd.DataFrame:
    """Weekly aggregated player stats (faster than full PBP for season features)."""
    import nfl_data_py as nfl
    try:
        return nfl.import_weekly_data(seasons)
    except Exception as exc:
        print(f"nfl_data_py weekly_data error: {exc}")
        return pd.DataFrame()


# ── Feature computation ────────────────────────────────────────────────────────

def schedules_to_game_logs(df: pd.DataFrame) -> list:
    """Convert schedules DataFrame to standard game log dicts."""
    if df.empty:
        return []
    logs = []
    for _, row in df.iterrows():
        if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
            continue
        home_score = float(row["home_score"])
        away_score = float(row["away_score"])
        logs.append({
            "game_id": str(row.get("game_id", "")),
            "game_date": str(pd.to_datetime(row.get("gameday", "")).date()) if pd.notna(row.get("gameday")) else "",
            "season": int(row.get("season", 0)),
            "week": int(row.get("week", 0)),
            "home_team": str(row.get("home_team", "")),
            "away_team": str(row.get("away_team", "")),
            "home_score": int(home_score),
            "away_score": int(away_score),
            "home_win": home_score > away_score,
            "home_rest_days": int(row["home_rest"]) if pd.notna(row.get("home_rest")) else 7,
            "away_rest_days": int(row["away_rest"]) if pd.notna(row.get("away_rest")) else 7,
            "weather_temp": float(row["temp"]) if pd.notna(row.get("temp")) else None,
            "weather_wind": float(row["wind"]) if pd.notna(row.get("wind")) else None,
            "surface": str(row.get("surface", "")),
            "roof": str(row.get("roof", "")),
            "stadium": str(row.get("stadium", "")),
            "referee": str(row.get("referee", "")),
            "source": "nfl_data_py",
        })
    return logs


def compute_epa_metrics(pbp_df: pd.DataFrame, team: str, n_games: int = 8) -> dict:
    """EPA-per-play as DVOA proxy for a team over recent games."""
    if pbp_df.empty or "epa" not in pbp_df.columns:
        return {"off_epa": 0.0, "def_epa": 0.0}
    max_plays = n_games * 70
    off = pbp_df[pbp_df["posteam"] == team]["epa"].tail(max_plays)
    def_ = pbp_df[pbp_df["defteam"] == team]["epa"].tail(max_plays)
    return {
        "off_epa": float(off.mean()) if not off.empty else 0.0,
        "def_epa": float(def_.mean()) if not def_.empty else 0.0,
    }


def compute_participation_rate_on_high_leverage(
    participation_df: pd.DataFrame,
    pbp_df: pd.DataFrame,
    team: str,
    n_games: int = 8,
) -> float:
    """
    Fraction of high-leverage plays (win probability 40-60%) where
    the team's starters are participating. Higher = more consistently deployed.
    """
    if participation_df.empty or pbp_df.empty:
        return 0.75
    if "wp" not in pbp_df.columns or "game_id" not in participation_df.columns:
        return 0.75
    try:
        # High-leverage: WP between 0.40 and 0.60 (competitive plays)
        hl_plays = pbp_df[
            (pbp_df.get("wp", pd.Series(0.5)).between(0.40, 0.60)) &
            ((pbp_df.get("posteam", pd.Series()) == team) |
             (pbp_df.get("defteam", pd.Series()) == team))
        ]["play_id" if "play_id" in pbp_df.columns else pbp_df.columns[0]].tail(n_games * 15)

        if hl_plays.empty:
            return 0.75

        team_part = participation_df[
            participation_df.get("possession_team", participation_df.get("team", pd.Series())) == team
        ]
        if team_part.empty:
            return 0.75

        # Unique players on these plays vs roster size
        players_on_hl = team_part[
            team_part.get("play_id", team_part.index).isin(hl_plays)
        ].get("gsis_id", pd.Series()).nunique()
        total_on_team = team_part.get("gsis_id", pd.Series()).nunique()
        if total_on_team == 0:
            return 0.75
        return float(min(1.0, players_on_hl / max(total_on_team * 0.5, 1)))
    except Exception:
        return 0.75


def compute_nextgen_separation(nextgen_df: pd.DataFrame, team: str) -> float:
    """Average receiver separation (yards) for a team from NextGen receiving stats."""
    if nextgen_df.empty:
        return 2.5
    col_team = "team_abbr" if "team_abbr" in nextgen_df.columns else "team"
    col_sep = "avg_separation" if "avg_separation" in nextgen_df.columns else None
    if not col_sep:
        return 2.5
    team_df = nextgen_df[nextgen_df.get(col_team, pd.Series()) == team]
    if team_df.empty:
        return 2.5
    return float(team_df[col_sep].mean())


def compute_nextgen_pass_time(nextgen_df: pd.DataFrame, team: str) -> float:
    """Average time to throw (seconds) for a team from NextGen passing stats."""
    if nextgen_df.empty:
        return 2.7
    col_team = "team_abbr" if "team_abbr" in nextgen_df.columns else "team"
    col_tt = "avg_time_to_throw" if "avg_time_to_throw" in nextgen_df.columns else None
    if not col_tt:
        return 2.7
    team_df = nextgen_df[nextgen_df.get(col_team, pd.Series()) == team]
    if team_df.empty:
        return 2.7
    return float(team_df[col_tt].mean())


def compute_epa_enriched(pbp_df: pd.DataFrame, game_logs: list) -> list:
    """
    Enrich game logs with pre-game EPA rolling averages.
    Mutates game_logs in-place with home_off_epa, home_def_epa, etc.
    """
    if pbp_df.empty:
        return game_logs
    team_epa: dict = {}
    for g in sorted(game_logs, key=lambda x: x.get("game_date", "")):
        h, a = g.get("home_team", ""), g.get("away_team", "")
        g["home_off_epa"] = team_epa.get((h, "off"), 0.0)
        g["home_def_epa"] = team_epa.get((h, "def"), 0.0)
        g["away_off_epa"] = team_epa.get((a, "off"), 0.0)
        g["away_def_epa"] = team_epa.get((a, "def"), 0.0)
        # Update rolling EPA from this game's PBP
        gid = g.get("game_id", "")
        game_pbp = pbp_df[pbp_df.get("game_id", pd.Series()) == gid] if gid else pd.DataFrame()
        if not game_pbp.empty and "epa" in game_pbp.columns:
            for team, side in [(h, "home"), (a, "away")]:
                off_epa = float(game_pbp[game_pbp["posteam"] == team]["epa"].mean() or 0.0)
                def_epa = float(game_pbp[game_pbp["defteam"] == team]["epa"].mean() or 0.0)
                old_off = team_epa.get((team, "off"), 0.0)
                old_def = team_epa.get((team, "def"), 0.0)
                # Exponential smoothing: alpha=0.3
                team_epa[(team, "off")] = 0.7 * old_off + 0.3 * off_epa
                team_epa[(team, "def")] = 0.7 * old_def + 0.3 * def_epa
    return game_logs


def get_injury_flags(injuries_df: pd.DataFrame, team: str, game_date: str) -> int:
    """Return 1 if team has any Questionable/Doubtful/Out players near game_date."""
    if injuries_df.empty:
        return 0
    bad_statuses = {"Questionable", "Doubtful", "Out", "IR"}
    col_team = "team" if "team" in injuries_df.columns else "team_abbr"
    col_status = "report_status" if "report_status" in injuries_df.columns else "injury_status"
    col_date = "week" if "week" in injuries_df.columns else "game_date"
    try:
        team_inj = injuries_df[
            (injuries_df.get(col_team, pd.Series()) == team) &
            (injuries_df.get(col_status, pd.Series()).isin(bad_statuses))
        ]
        return int(not team_inj.empty)
    except Exception:
        return 0


def compute_cap_spending(contracts_df: pd.DataFrame, team: str) -> float:
    """Team's total guaranteed cap spending as fraction of league avg (0.5-1.5)."""
    if contracts_df.empty:
        return 1.0
    col_team = "team" if "team" in contracts_df.columns else "team_abbr"
    col_value = "value" if "value" in contracts_df.columns else "apy_cap_pct"
    try:
        team_df = contracts_df[contracts_df.get(col_team, pd.Series()) == team]
        if team_df.empty or col_value not in team_df.columns:
            return 1.0
        team_avg = float(team_df[col_value].sum())
        league_avg = float(contracts_df.groupby(col_team)[col_value].sum().mean())
        return round(team_avg / league_avg, 4) if league_avg > 0 else 1.0
    except Exception:
        return 1.0


# ── Full history ingestion ─────────────────────────────────────────────────────

def ingest_full_history(seasons: list, incremental_from=None,
                        pull_nextgen: bool = True,
                        pull_injuries: bool = True,
                        pull_contracts: bool = True) -> list:
    """
    Master NFL ingestion: schedules → PBP EPA → participation → NextGen → injuries.
    Returns enriched game log dicts with all features.
    Downloads large parquet files; plan for 1-2 hours.
    """
    print(f"[NFL] Ingesting {len(seasons)} seasons ({seasons[0]}-{seasons[-1]})...")

    # Schedules (primary game log source)
    sched_df = get_schedules(seasons)
    game_logs = schedules_to_game_logs(sched_df)
    print(f"[NFL] Schedules: {len(game_logs)} completed games")

    if incremental_from:
        game_logs = [g for g in game_logs if g.get("game_date", "") > str(incremental_from)]

    # PBP: full EPA computation
    print("[NFL] Downloading play-by-play (large parquet)...")
    pbp_df = get_pbp_slim(seasons)
    print(f"[NFL] PBP: {len(pbp_df):,} rows")
    if not pbp_df.empty:
        game_logs = compute_epa_enriched(pbp_df, game_logs)
        print("[NFL] EPA metrics enriched")

    # Participation: high-leverage play rates
    print("[NFL] Downloading participation data (~50M rows, may take ~20min)...")
    try:
        part_df = get_participation(seasons)
        print(f"[NFL] Participation: {len(part_df):,} rows")
        # Compute per-team participation rates and add to game logs
        for g in game_logs:
            h, a = g.get("home_team", ""), g.get("away_team", "")
            try:
                game_pbp = pbp_df[pbp_df.get("game_id", pd.Series()) == g.get("game_id", "")] if not pbp_df.empty else pd.DataFrame()
                g["home_participation_rate"] = compute_participation_rate_on_high_leverage(
                    part_df, game_pbp, h)
                g["away_participation_rate"] = compute_participation_rate_on_high_leverage(
                    part_df, game_pbp, a)
            except Exception:
                g["home_participation_rate"] = 0.75
                g["away_participation_rate"] = 0.75
    except Exception as exc:
        print(f"[NFL] Participation error: {exc}")

    # NextGen receiving + passing stats
    if pull_nextgen:
        print("[NFL] Downloading NextGen receiving stats...")
        try:
            ng_recv = get_nextgen_stats(seasons, "receiving")
            print(f"[NFL] NextGen receiving: {len(ng_recv):,} rows")
            team_separation = {}
            if not ng_recv.empty:
                col_team = "team_abbr" if "team_abbr" in ng_recv.columns else "team"
                if "avg_separation" in ng_recv.columns:
                    team_separation = ng_recv.groupby(col_team)["avg_separation"].mean().to_dict()
            for g in game_logs:
                g["home_nextgen_separation"] = float(team_separation.get(g.get("home_team", ""), 2.5))
                g["away_nextgen_separation"] = float(team_separation.get(g.get("away_team", ""), 2.5))
        except Exception as exc:
            print(f"[NFL] NextGen receiving error: {exc}")

        print("[NFL] Downloading NextGen passing stats...")
        try:
            ng_pass = get_nextgen_stats(seasons, "passing")
            team_ttt = {}
            if not ng_pass.empty:
                col_team = "team_abbr" if "team_abbr" in ng_pass.columns else "team"
                if "avg_time_to_throw" in ng_pass.columns:
                    team_ttt = ng_pass.groupby(col_team)["avg_time_to_throw"].mean().to_dict()
            for g in game_logs:
                g["home_time_to_throw"] = float(team_ttt.get(g.get("home_team", ""), 2.7))
                g["away_time_to_throw"] = float(team_ttt.get(g.get("away_team", ""), 2.7))
        except Exception as exc:
            print(f"[NFL] NextGen passing error: {exc}")
    else:
        print("[NFL] NextGen stats skipped")

    # Injuries
    if pull_injuries:
        print("[NFL] Downloading injury data...")
        try:
            inj_df = get_injuries(seasons)
            for g in game_logs:
                g["home_injury_flag"] = get_injury_flags(inj_df, g.get("home_team", ""), g.get("game_date", ""))
                g["away_injury_flag"] = get_injury_flags(inj_df, g.get("away_team", ""), g.get("game_date", ""))
        except Exception as exc:
            print(f"[NFL] Injury data error: {exc}")
    else:
        print("[NFL] Injuries skipped")

    # Contracts: cap spending ratio
    if pull_contracts:
        print("[NFL] Downloading contract data...")
        try:
            contracts_df = get_contracts()
            if not contracts_df.empty:
                for g in game_logs:
                    g["home_cap_ratio"] = compute_cap_spending(contracts_df, g.get("home_team", ""))
                    g["away_cap_ratio"] = compute_cap_spending(contracts_df, g.get("away_team", ""))
        except Exception as exc:
            print(f"[NFL] Contracts error: {exc}")
    else:
        print("[NFL] Contracts skipped")

    print(f"[NFL] Ingestion complete: {len(game_logs)} enriched game logs")
    return game_logs
