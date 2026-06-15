"""
data/scrapers/nba.py — Full NBA historical data via nba_api.
Pulls play-by-play, shot charts, player tracking, 5-man lineup stats,
team season stats, box scores, rest days, back-to-back flags.
Rate limit: 1 request / 0.65s.
"""
import time
import traceback
from datetime import date
from typing import Optional

import pandas as pd

from config import MAX_RETRIES, BACKOFF_BASE

_NBA_DELAY = 0.65  # seconds between nba_api calls


def _call(fn, *args, **kwargs):
    """Rate-limited wrapper with retries."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            result = fn(*args, **kwargs)
            time.sleep(_NBA_DELAY)
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
    raise last_exc


def _get_df(result, index: int = 0) -> "pd.DataFrame":
    """
    Extract DataFrame from nba_api result, tolerating resultSet vs resultSets
    key differences across nba_api versions and older game data.
    """
    import pandas as pd
    try:
        frames = result.get_data_frames()
        if frames and index < len(frames):
            return frames[index]
        return pd.DataFrame()
    except KeyError:
        # nba_api version mismatch: try raw resultSets
        try:
            raw = result.get_dict()
            rs = raw.get("resultSets") or raw.get("resultSet") or []
            if isinstance(rs, list) and index < len(rs):
                headers = rs[index].get("headers", [])
                rows = rs[index].get("rowSet", [])
                return pd.DataFrame(rows, columns=headers)
        except Exception:
            pass
        return pd.DataFrame()


def _season_str(year: int) -> str:
    """Convert season start year to NBA API format: 2023 → '2023-24'."""
    return f"{year}-{str(year + 1)[-2:]}"


# ── Game log retrieval ─────────────────────────────────────────────────────────

def get_season_games(season: int) -> list:
    """All regular-season games for a season year via LeagueGameFinder."""
    from nba_api.stats.endpoints import LeagueGameFinder
    season_str = _season_str(season)
    finder = _call(LeagueGameFinder,
                   season_nullable=season_str,
                   league_id_nullable="00",
                   season_type_nullable="Regular Season")
    df = _get_df(finder)

    games: dict = {}
    for _, row in df.iterrows():
        gid = str(row["GAME_ID"])
        matchup = str(row.get("MATCHUP", ""))
        is_home = "@" not in matchup
        if gid not in games:
            games[gid] = {"game_id": gid, "game_date": str(row["GAME_DATE"]),
                          "season": season, "source": "nba_api"}
        side = "home" if is_home else "away"
        games[gid][f"{side}_team"] = row["TEAM_ABBREVIATION"]
        games[gid][f"{side}_score"] = int(row.get("PTS") or 0)
        games[gid][f"{side}_win"] = row.get("WL") == "W"
        games[gid][f"{side}_team_id"] = int(row.get("TEAM_ID") or 0)

    result = [g for g in games.values() if "home_team" in g and "away_team" in g]
    for g in result:
        g["home_win"] = bool(g.get("home_win", False))
    return result


def add_rest_and_b2b(games: list) -> list:
    """Compute rest_days and back_to_back per team per game (in-place)."""
    last: dict = {}
    for g in sorted(games, key=lambda x: x["game_date"]):
        gdate = pd.to_datetime(g["game_date"]).date()
        for side in ("home", "away"):
            team = g.get(f"{side}_team")
            if not team:
                continue
            if team in last:
                rest = (gdate - last[team]).days
                g[f"{side}_rest_days"] = rest
                g[f"{side}_back_to_back"] = rest <= 1
            else:
                g[f"{side}_rest_days"] = 7
                g[f"{side}_back_to_back"] = False
            last[team] = gdate
    return games


# ── Play-by-play ───────────────────────────────────────────────────────────────

def get_play_by_play(game_id: str) -> pd.DataFrame:
    """Raw play-by-play for one game via PlayByPlayV2."""
    from nba_api.stats.endpoints import PlayByPlayV2
    try:
        result = _call(PlayByPlayV2, game_id=game_id)
        return _get_df(result)
    except Exception as exc:
        print(f"PlayByPlayV2 error {game_id}: {exc}")
        return pd.DataFrame()


def compute_pace_from_pbp(pbp_df: pd.DataFrame) -> dict:
    """
    Estimate possessions (pace) from play-by-play.
    Possession ends on: made FG, made last FT, turnover, defensive rebound.
    """
    if pbp_df.empty or "EVENTMSGTYPE" not in pbp_df.columns:
        return {"possessions": 180, "pace": 98.0}
    # EVENTMSGTYPE: 1=made FG, 2=missed FG, 3=free throw, 4=rebound, 5=turnover
    poss_events = pbp_df[pbp_df["EVENTMSGTYPE"].isin([1, 5])].shape[0]
    # Add defensive rebounds (rebound after missed FG by opponent)
    rebounds = pbp_df[pbp_df["EVENTMSGTYPE"] == 4]
    def_reb = rebounds[rebounds.get("PLAYER1_TEAM_ID", rebounds.get("PERSON1TYPE", pd.Series())).notna()].shape[0]
    total_poss = int(poss_events + def_reb * 0.5)
    total_poss = max(total_poss, 80)
    minutes = 48.0
    pace = (total_poss / minutes) * 48.0
    return {"possessions": total_poss, "pace": round(pace, 2)}


def compute_referee_foul_rate(pbp_df: pd.DataFrame) -> float:
    """Fouls per possession from play-by-play (EVENTMSGTYPE=6)."""
    if pbp_df.empty or "EVENTMSGTYPE" not in pbp_df.columns:
        return 0.20
    fouls = pbp_df[pbp_df["EVENTMSGTYPE"] == 6].shape[0]
    total = max(pbp_df.shape[0], 1)
    return round(fouls / total, 4)


# ── Shot charts ────────────────────────────────────────────────────────────────

def get_shot_chart(game_id: str, team_id: int, season: str) -> pd.DataFrame:
    """Shot locations for one team in one game via ShotChartDetail."""
    from nba_api.stats.endpoints import ShotChartDetail
    try:
        result = _call(ShotChartDetail,
                       player_id=0,
                       team_id=team_id,
                       game_id_nullable=game_id,
                       season_nullable=season,
                       season_type_all_star="Regular Season")
        return _get_df(result)
    except Exception as exc:
        print(f"ShotChartDetail error {game_id} team {team_id}: {exc}")
        return pd.DataFrame()


def compute_shot_zone_efficiency(shot_df: pd.DataFrame) -> dict:
    """
    eFG% by zone: restricted area, mid-range, three-point.
    Returns {ra_efg, mid_efg, three_efg, overall_efg}.
    """
    if shot_df.empty or "SHOT_MADE_FLAG" not in shot_df.columns:
        return {"ra_efg": 0.63, "mid_efg": 0.40, "three_efg": 0.53, "overall_efg": 0.52}

    def efg(df_zone, multiplier=1.0):
        made = df_zone["SHOT_MADE_FLAG"].sum()
        att = len(df_zone)
        return (made + multiplier * 0) / att if att > 0 else 0.0

    zone_col = "SHOT_ZONE_BASIC" if "SHOT_ZONE_BASIC" in shot_df.columns else None
    if zone_col:
        ra = shot_df[shot_df[zone_col] == "Restricted Area"]
        mid = shot_df[shot_df[zone_col].isin(["Mid-Range", "In The Paint (Non-RA)"])]
        three = shot_df[shot_df[zone_col].isin(["Above the Break 3", "Left Corner 3", "Right Corner 3"])]
    else:
        ra = shot_df[shot_df.get("SHOT_DISTANCE", pd.Series()) <= 4]
        three = shot_df[shot_df.get("SHOT_TYPE", pd.Series()).str.contains("3PT", na=False)]
        mid = shot_df[~shot_df.index.isin(ra.index) & ~shot_df.index.isin(three.index)]

    total_made = shot_df["SHOT_MADE_FLAG"].sum()
    three_made = three["SHOT_MADE_FLAG"].sum() if not three.empty else 0
    total_att = len(shot_df)
    overall_efg = (total_made + 0.5 * three_made) / total_att if total_att > 0 else 0.52

    return {
        "ra_efg": round(efg(ra), 4),
        "mid_efg": round(efg(mid), 4),
        "three_efg": round(efg(three), 4),
        "overall_efg": round(overall_efg, 4),
    }


# ── Player tracking ────────────────────────────────────────────────────────────

def get_player_tracking(team_id: int, season: str) -> pd.DataFrame:
    """Speed, distance, touches per game for all players on a team (season avg)."""
    from nba_api.stats.endpoints import PlayerDashPtStats
    try:
        result = _call(PlayerDashPtStats, team_id=str(team_id), season=season,
                       per_mode="PerGame", season_type_all_star="Regular Season")
        return _get_df(result)
    except Exception as exc:
        print(f"PlayerDashPtStats error team {team_id} {season}: {exc}")
        return pd.DataFrame()


def compute_tracking_metrics(tracking_df: pd.DataFrame) -> dict:
    """
    Aggregate tracking metrics to team level.
    Returns {avg_speed, avg_distance, avg_touches, avg_drives}.
    """
    if tracking_df.empty:
        return {"avg_speed": 4.5, "avg_distance": 2.8, "avg_touches": 60.0, "avg_drives": 15.0}
    cols = {
        "avg_speed": "AVG_SPEED" if "AVG_SPEED" in tracking_df.columns else None,
        "avg_distance": "DIST_MILES" if "DIST_MILES" in tracking_df.columns else None,
        "avg_touches": "TOUCHES" if "TOUCHES" in tracking_df.columns else None,
        "avg_drives": "DRIVES" if "DRIVES" in tracking_df.columns else None,
    }
    result = {}
    for key, col in cols.items():
        if col and col in tracking_df.columns:
            result[key] = float(tracking_df[col].mean())
        else:
            result[key] = {"avg_speed": 4.5, "avg_distance": 2.8,
                           "avg_touches": 60.0, "avg_drives": 15.0}[key]
    return result


# ── 5-man lineups ──────────────────────────────────────────────────────────────

def get_lineup_stats(team_id: int, season: str) -> pd.DataFrame:
    """All 5-man lineup combinations and net ratings for a team (full season)."""
    from nba_api.stats.endpoints import TeamDashLineups
    try:
        result = _call(TeamDashLineups, team_id=str(team_id), season=season,
                       measure_type_detailed_defense="Advanced",
                       per_mode_simple="Per100Possessions",
                       season_type_all_star="Regular Season")
        return _get_df(result)
    except Exception as exc:
        print(f"TeamDashLineups error team {team_id} {season}: {exc}")
        return pd.DataFrame()


def compute_best_lineup_net_rating(lineup_df: pd.DataFrame, min_poss: int = 50) -> float:
    """Net rating (off_rtg - def_rtg) of the most-used qualifying 5-man lineup."""
    if lineup_df.empty:
        return 0.0
    needed = {"OFF_RATING", "DEF_RATING", "MIN"}
    if not needed.issubset(lineup_df.columns):
        return 0.0
    lineup_df = lineup_df.copy()
    lineup_df["NET_RATING"] = lineup_df["OFF_RATING"] - lineup_df["DEF_RATING"]
    # Filter by minimum possessions proxy (MIN > threshold)
    qualified = lineup_df[lineup_df["MIN"] >= min_poss]
    if qualified.empty:
        qualified = lineup_df
    # Top by minutes played
    top = qualified.sort_values("MIN", ascending=False).iloc[0]
    return float(top["NET_RATING"])


# ── Team season stats ──────────────────────────────────────────────────────────

def get_team_season_stats(season: str) -> pd.DataFrame:
    """Advanced team stats (pace, off_rtg, def_rtg, etc.) for a full season."""
    from nba_api.stats.endpoints import LeagueDashTeamStats
    try:
        result = _call(LeagueDashTeamStats,
                       season=season,
                       measure_type_detailed_defense="Advanced",
                       per_mode_simple="PerGame",
                       season_type_all_star="Regular Season")
        return _get_df(result)
    except Exception as exc:
        print(f"LeagueDashTeamStats error {season}: {exc}")
        return pd.DataFrame()


def extract_team_advanced(stats_df: pd.DataFrame, team_abbr: str) -> dict:
    """Extract advanced stats for a specific team from LeagueDashTeamStats."""
    if stats_df.empty:
        return {}
    team_row = stats_df[stats_df.get("TEAM_ABBREVIATION", stats_df.get("TEAM", pd.Series())) == team_abbr]
    if team_row.empty:
        return {}
    row = team_row.iloc[0]
    return {
        "off_rtg": float(row.get("OFF_RATING") or 110.0),
        "def_rtg": float(row.get("DEF_RATING") or 110.0),
        "pace": float(row.get("PACE") or 98.0),
        "ts_pct": float(row.get("TS_PCT") or 0.55),
        "ast_pct": float(row.get("AST_PCT") or 0.60),
    }


# ── Advanced box score ─────────────────────────────────────────────────────────

def get_advanced_stats(game_id: str) -> dict:
    """TS% and USG% (starter averages) for home and away teams."""
    try:
        from nba_api.stats.endpoints import BoxScoreAdvancedV2
        adv = _call(BoxScoreAdvancedV2, game_id=game_id)
        df = _get_df(adv)
        starters = df[df["START_POSITION"].notna() & (df["START_POSITION"] != "")]
        if starters.empty:
            return {}
        teams = starters["TEAM_ID"].unique()
        if len(teams) < 2:
            return {}
        result = {}
        for i, (side, tid) in enumerate(zip(("home", "away"), teams)):
            t_df = starters[starters["TEAM_ID"] == tid]
            result[f"{side}_ts_pct"] = float(t_df["TS_PCT"].mean() if "TS_PCT" in t_df.columns else 0.55)
            result[f"{side}_usg_pct"] = float(t_df["USG_PCT"].mean() if "USG_PCT" in t_df.columns else 0.20)
        return result
    except Exception as exc:
        print(f"BoxScoreAdvancedV2 error {game_id}: {exc}")
        return {}


def get_officials(game_id: str) -> str:
    """Return comma-joined referee names for a game."""
    try:
        from nba_api.stats.endpoints import BoxScoreSummaryV2
        summary = _call(BoxScoreSummaryV2, game_id=game_id)
        officials_df = _get_df(summary, 2)
        if officials_df.empty:
            return ""
        names = officials_df.apply(
            lambda r: f"{r.get('FIRST_NAME','')} {r.get('LAST_NAME','')}".strip(), axis=1
        )
        return ",".join(names.tolist())
    except Exception:
        return ""


# ── Referee tendencies ─────────────────────────────────────────────────────────

def compute_referee_tendencies(all_game_pbp: list) -> dict:
    """
    Build referee tendency table from list of (game_id, referee, pbp_df) tuples.
    Returns {referee: {pace_factor, foul_rate, games_officiated}}.
    """
    ref_data: dict = {}
    for game_id, referee, pbp_df in all_game_pbp:
        if not referee:
            continue
        pace_info = compute_pace_from_pbp(pbp_df)
        foul_rate = compute_referee_foul_rate(pbp_df)
        for ref in referee.split(","):
            ref = ref.strip()
            if not ref:
                continue
            if ref not in ref_data:
                ref_data[ref] = {"pace_total": 0.0, "foul_total": 0.0, "count": 0}
            ref_data[ref]["pace_total"] += pace_info.get("pace", 98.0)
            ref_data[ref]["foul_total"] += foul_rate
            ref_data[ref]["count"] += 1
    result = {}
    for ref, data in ref_data.items():
        n = data["count"]
        if n < 5:
            continue
        result[ref] = {
            "pace_factor": round(data["pace_total"] / n / 98.0, 4),
            "foul_rate": round(data["foul_total"] / n, 4),
            "games_officiated": n,
        }
    return result


# ── Kaggle dataset conversion ──────────────────────────────────────────────────

def kaggle_to_game_logs(df: pd.DataFrame) -> list:
    """Convert nathanlauga/nba-games Kaggle DataFrame to standard game log dicts."""
    if df.empty:
        return []
    logs = []
    for _, row in df.iterrows():
        try:
            home_win = bool(row.get("HOME_TEAM_WINS", row.get("home_team_wins", 1)))
            logs.append({
                "game_id": f"kaggle_nba_{row.get('GAME_ID', row.get('game_id', ''))}",
                "game_date": str(row.get("GAME_DATE_EST", row.get("game_date", "")))[:10],
                "season": int(row.get("SEASON", 0)),
                "home_team": str(row.get("HOME_TEAM_ID", row.get("home_team", ""))),
                "away_team": str(row.get("VISITOR_TEAM_ID", row.get("away_team", ""))),
                "home_score": int(float(row.get("PTS_home", row.get("home_score", 0)) or 0)),
                "away_score": int(float(row.get("PTS_away", row.get("away_score", 0)) or 0)),
                "home_win": home_win,
                "source": "kaggle_nba",
            })
        except Exception:
            continue
    return [g for g in logs if g["game_date"] and g["home_team"] and g["away_team"]]


def wyattowalsh_to_game_logs(df: pd.DataFrame) -> list:
    """Convert wyattowalsh/basketball Kaggle DataFrame to game log dicts."""
    if df.empty:
        return []
    logs = []
    needed = {"game_id", "game_date", "home_team_abbreviation", "away_team_abbreviation",
              "home_pts", "away_pts"}
    if not needed.issubset(set(df.columns)):
        return []
    for _, row in df.iterrows():
        try:
            h_pts = int(float(row.get("home_pts") or 0))
            a_pts = int(float(row.get("away_pts") or 0))
            logs.append({
                "game_id": f"wyatto_{row['game_id']}",
                "game_date": str(row["game_date"])[:10],
                "season": int(str(row.get("season", "0")).split("-")[0]),
                "home_team": str(row["home_team_abbreviation"]),
                "away_team": str(row["away_team_abbreviation"]),
                "home_score": h_pts,
                "away_score": a_pts,
                "home_win": h_pts > a_pts,
                "source": "kaggle_wyattowalsh",
            })
        except Exception:
            continue
    return [g for g in logs if g["game_date"] and g["home_team"] and g["away_team"]]


# ── Full history ingestion ─────────────────────────────────────────────────────

def ingest_full_history(seasons: list, incremental_from=None,
                        pull_pbp: bool = True,
                        pull_shots: bool = True,
                        pull_lineups: bool = True) -> list:
    """
    Master ingestion for all NBA seasons.
    For each season: game logs → rest days → PBP → shot charts → lineups → advanced box.
    Stores aggregated metrics in game log dicts. Raw PBP is processed in memory only.
    Returns combined list of enriched game log dicts.
    """
    from nba_api.stats.static import teams as nba_teams_static
    all_logs = []
    all_teams = {t["abbreviation"]: t["id"] for t in nba_teams_static.get_teams()}

    for season in seasons:
        season_str = _season_str(season)
        print(f"[NBA] Ingesting season {season_str}...")
        try:
            games = get_season_games(season)
            games = add_rest_and_b2b(games)
        except Exception as exc:
            print(f"[NBA] Failed to get game list for {season_str}: {exc}")
            continue

        if incremental_from:
            games = [g for g in games if g.get("game_date", "") > str(incremental_from)]

        if not games:
            continue

        # Season-level: team advanced stats (one call per season)
        team_advanced = {}
        try:
            season_stats_df = get_team_season_stats(season_str)
            for abbr in all_teams:
                team_advanced[abbr] = extract_team_advanced(season_stats_df, abbr)
        except Exception as exc:
            print(f"[NBA] Season stats error {season_str}: {exc}")

        # Season-level: lineup net ratings per team (N_teams × 1 call each)
        lineup_net_ratings = {}
        if pull_lineups:
            for abbr, team_id in all_teams.items():
                try:
                    lu_df = get_lineup_stats(team_id, season_str)
                    lineup_net_ratings[abbr] = compute_best_lineup_net_rating(lu_df)
                except Exception:
                    lineup_net_ratings[abbr] = 0.0

        # Season-level: player tracking per team
        tracking_metrics = {}
        for abbr, team_id in all_teams.items():
            try:
                tr_df = get_player_tracking(team_id, season_str)
                tracking_metrics[abbr] = compute_tracking_metrics(tr_df)
            except Exception:
                tracking_metrics[abbr] = {}

        # Game-level: PBP, shot chart, advanced box
        pbp_cache = {}
        for g in games:
            game_id = g["game_id"]
            h_team = g.get("home_team", "")
            a_team = g.get("away_team", "")
            h_team_id = g.get("home_team_id", all_teams.get(h_team, 0))
            a_team_id = g.get("away_team_id", all_teams.get(a_team, 0))

            # Play-by-play → pace + foul rate
            if pull_pbp:
                try:
                    pbp_df = get_play_by_play(game_id)
                    pace_info = compute_pace_from_pbp(pbp_df)
                    g["home_pace"] = pace_info.get("pace", 98.0)
                    g["away_pace"] = pace_info.get("pace", 98.0)
                    g["game_foul_rate"] = compute_referee_foul_rate(pbp_df)
                    pbp_cache[game_id] = pbp_df
                except Exception:
                    g["home_pace"] = team_advanced.get(h_team, {}).get("pace", 98.0)
                    g["away_pace"] = team_advanced.get(a_team, {}).get("pace", 98.0)
                    g["game_foul_rate"] = 0.20

            # Shot chart → eFG%
            if pull_shots and h_team_id:
                try:
                    h_shots = get_shot_chart(game_id, h_team_id, season_str)
                    a_shots = get_shot_chart(game_id, a_team_id, season_str)
                    h_eff = compute_shot_zone_efficiency(h_shots)
                    a_eff = compute_shot_zone_efficiency(a_shots)
                    g["home_efg"] = h_eff.get("overall_efg", 0.52)
                    g["away_efg"] = a_eff.get("overall_efg", 0.52)
                    g["home_three_efg"] = h_eff.get("three_efg", 0.37)
                    g["away_three_efg"] = a_eff.get("three_efg", 0.37)
                except Exception:
                    g["home_efg"] = 0.52
                    g["away_efg"] = 0.52

            # Advanced box score → TS%, USG%
            try:
                adv = get_advanced_stats(game_id)
                g.update(adv)
            except Exception:
                pass

            # Officials
            try:
                g["referee"] = get_officials(game_id)
            except Exception:
                g["referee"] = ""

            # Lineup net ratings (season-level, not per-game)
            g["home_lineup_net_rating"] = lineup_net_ratings.get(h_team, 0.0)
            g["away_lineup_net_rating"] = lineup_net_ratings.get(a_team, 0.0)

            # Advanced season stats
            for side, team in (("home", h_team), ("away", a_team)):
                adv_s = team_advanced.get(team, {})
                g[f"{side}_off_rtg"] = adv_s.get("off_rtg", 110.0)
                g[f"{side}_def_rtg"] = adv_s.get("def_rtg", 110.0)

            # Tracking
            for side, team in (("home", h_team), ("away", a_team)):
                tr = tracking_metrics.get(team, {})
                g[f"{side}_avg_speed"] = tr.get("avg_speed", 4.5)
                g[f"{side}_avg_drives"] = tr.get("avg_drives", 15.0)

        print(f"[NBA] Season {season_str}: {len(games)} games enriched")
        all_logs.extend(games)

    return all_logs
