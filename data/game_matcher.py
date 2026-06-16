"""
data/game_matcher.py — Fuzzy-match prediction market titles to Supabase game records.
Uses rapidfuzz with 0.85 minimum confidence threshold.
"""
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from rapidfuzz import fuzz, process

MATCH_THRESHOLD = 85.0

NBA_TEAMS = {
    "Atlanta Hawks": ["Atlanta", "Hawks", "ATL"],
    "Boston Celtics": ["Boston", "Celtics", "BOS"],
    "Brooklyn Nets": ["Brooklyn", "Nets", "BKN", "NJN"],
    "Charlotte Hornets": ["Charlotte", "Hornets", "CHA"],
    "Chicago Bulls": ["Chicago Bulls", "Bulls", "CHI"],
    "Cleveland Cavaliers": ["Cleveland", "Cavaliers", "Cavs", "CLE"],
    "Dallas Mavericks": ["Dallas", "Mavericks", "Mavs", "DAL"],
    "Denver Nuggets": ["Denver", "Nuggets", "DEN"],
    "Detroit Pistons": ["Detroit", "Pistons", "DET"],
    "Golden State Warriors": ["Golden State", "Warriors", "GSW"],
    "Houston Rockets": ["Houston Rockets", "Rockets", "HOU"],
    "Indiana Pacers": ["Indiana", "Pacers", "IND"],
    "LA Clippers": ["LA Clippers", "Clippers", "LAC"],
    "Los Angeles Lakers": ["Los Angeles Lakers", "Lakers", "LAL"],
    "Memphis Grizzlies": ["Memphis", "Grizzlies", "MEM"],
    "Miami Heat": ["Miami Heat", "Heat", "MIA"],
    "Milwaukee Bucks": ["Milwaukee", "Bucks", "MIL"],
    "Minnesota Timberwolves": ["Minnesota", "Timberwolves", "Wolves", "MIN"],
    "New Orleans Pelicans": ["New Orleans", "Pelicans", "NOP"],
    "New York Knicks": ["New York Knicks", "Knicks", "NYK"],
    "Oklahoma City Thunder": ["Oklahoma City", "Thunder", "OKC"],
    "Orlando Magic": ["Orlando", "Magic", "ORL"],
    "Philadelphia 76ers": ["Philadelphia", "76ers", "Sixers", "PHI"],
    "Phoenix Suns": ["Phoenix", "Suns", "PHX"],
    "Portland Trail Blazers": ["Portland", "Trail Blazers", "Blazers", "POR"],
    "Sacramento Kings": ["Sacramento", "Kings", "SAC"],
    "San Antonio Spurs": ["San Antonio", "Spurs", "SAS"],
    "Toronto Raptors": ["Toronto", "Raptors", "TOR"],
    "Utah Jazz": ["Utah", "Jazz", "UTA"],
    "Washington Wizards": ["Washington Wizards", "Wizards", "WAS"],
}

NFL_TEAMS = {
    "Arizona Cardinals": ["Arizona", "Cardinals", "ARI"],
    "Atlanta Falcons": ["Atlanta Falcons", "Falcons"],
    "Baltimore Ravens": ["Baltimore", "Ravens", "BAL"],
    "Buffalo Bills": ["Buffalo", "Bills", "BUF"],
    "Carolina Panthers": ["Carolina", "Panthers", "CAR"],
    "Chicago Bears": ["Chicago Bears", "Bears"],
    "Cincinnati Bengals": ["Cincinnati", "Bengals", "CIN"],
    "Cleveland Browns": ["Cleveland Browns", "Browns"],
    "Dallas Cowboys": ["Dallas Cowboys", "Cowboys"],
    "Denver Broncos": ["Denver Broncos", "Broncos"],
    "Detroit Lions": ["Detroit Lions", "Lions"],
    "Green Bay Packers": ["Green Bay", "Packers", "GB"],
    "Houston Texans": ["Houston Texans", "Texans"],
    "Indianapolis Colts": ["Indianapolis", "Colts", "IND"],
    "Jacksonville Jaguars": ["Jacksonville", "Jaguars", "Jags", "JAX"],
    "Kansas City Chiefs": ["Kansas City", "Chiefs", "KC"],
    "Las Vegas Raiders": ["Las Vegas", "Raiders", "LV", "Oakland Raiders"],
    "Los Angeles Chargers": ["LA Chargers", "Chargers", "LAC"],
    "Los Angeles Rams": ["LA Rams", "Rams", "LAR"],
    "Miami Dolphins": ["Miami Dolphins", "Dolphins"],
    "Minnesota Vikings": ["Minnesota Vikings", "Vikings"],
    "New England Patriots": ["New England", "Patriots", "NE"],
    "New Orleans Saints": ["New Orleans Saints", "Saints"],
    "New York Giants": ["NY Giants", "Giants", "NYG"],
    "New York Jets": ["NY Jets", "Jets", "NYJ"],
    "Philadelphia Eagles": ["Philadelphia Eagles", "Eagles"],
    "Pittsburgh Steelers": ["Pittsburgh", "Steelers", "PIT"],
    "San Francisco 49ers": ["San Francisco", "49ers", "Niners", "SF"],
    "Seattle Seahawks": ["Seattle", "Seahawks", "SEA"],
    "Tampa Bay Buccaneers": ["Tampa Bay", "Buccaneers", "Bucs", "TB"],
    "Tennessee Titans": ["Tennessee", "Titans", "TEN"],
    "Washington Commanders": ["Washington Commanders", "Commanders", "Redskins", "WAS"],
}

MLB_TEAMS = {
    "Arizona Diamondbacks": ["Arizona Diamondbacks", "Diamondbacks", "D-backs", "ARI"],
    "Atlanta Braves": ["Atlanta Braves", "Braves"],
    "Baltimore Orioles": ["Baltimore Orioles", "Orioles", "BAL"],
    "Boston Red Sox": ["Boston Red Sox", "Red Sox", "BOS"],
    "Chicago Cubs": ["Chicago Cubs", "Cubs", "CHC"],
    "Chicago White Sox": ["Chicago White Sox", "White Sox", "CWS"],
    "Cincinnati Reds": ["Cincinnati Reds", "Reds", "CIN"],
    "Cleveland Guardians": ["Cleveland Guardians", "Guardians", "Indians", "CLE"],
    "Colorado Rockies": ["Colorado Rockies", "Rockies", "COL"],
    "Detroit Tigers": ["Detroit Tigers", "Tigers", "DET"],
    "Houston Astros": ["Houston Astros", "Astros"],
    "Kansas City Royals": ["Kansas City Royals", "Royals", "KC"],
    "Los Angeles Angels": ["LA Angels", "Angels", "LAA", "Anaheim Angels"],
    "Los Angeles Dodgers": ["LA Dodgers", "Dodgers", "LAD"],
    "Miami Marlins": ["Miami Marlins", "Marlins", "MIA"],
    "Milwaukee Brewers": ["Milwaukee Brewers", "Brewers", "MIL"],
    "Minnesota Twins": ["Minnesota Twins", "Twins", "MIN"],
    "New York Mets": ["NY Mets", "Mets", "NYM"],
    "New York Yankees": ["NY Yankees", "Yankees", "NYY"],
    "Oakland Athletics": ["Oakland Athletics", "Athletics", "OAK"],
    "Philadelphia Phillies": ["Philadelphia Phillies", "Phillies"],
    "Pittsburgh Pirates": ["Pittsburgh Pirates", "Pirates"],
    "San Diego Padres": ["San Diego Padres", "Padres", "SD"],
    "San Francisco Giants": ["San Francisco Giants", "SF Giants"],
    "Seattle Mariners": ["Seattle Mariners", "Mariners", "SEA"],
    "St. Louis Cardinals": ["St. Louis Cardinals", "Cardinals", "STL"],
    "Tampa Bay Rays": ["Tampa Bay Rays", "Rays", "TB"],
    "Texas Rangers": ["Texas Rangers", "Rangers", "TEX"],
    "Toronto Blue Jays": ["Toronto Blue Jays", "Blue Jays", "TOR"],
    "Washington Nationals": ["Washington Nationals", "Nationals", "Nats", "WSN"],
}

_ALIAS_MAP: dict = {}
_SPORT_TEAMS: dict = {
    "NBA": set(NBA_TEAMS.keys()),
    "NFL": set(NFL_TEAMS.keys()),
    "MLB": set(MLB_TEAMS.keys()),
}
for _sport, _teams in [("NBA", NBA_TEAMS), ("NFL", NFL_TEAMS), ("MLB", MLB_TEAMS)]:
    for _canonical, _aliases in _teams.items():
        _ALIAS_MAP[_canonical.lower()] = (_canonical, _sport)
        for _alias in _aliases:
            if _alias.lower() not in _ALIAS_MAP:
                _ALIAS_MAP[_alias.lower()] = (_canonical, _sport)

_ALL_ALIASES = list(_ALIAS_MAP.keys())


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(str(ts)[:19], fmt[:19])
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _resolve_team(raw: str) -> Optional[tuple]:
    raw = raw.strip()
    if not raw or len(raw) < 2:
        return None
    key = raw.lower()
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]
    match = process.extractOne(key, _ALL_ALIASES, scorer=fuzz.WRatio)
    if match and match[1] >= MATCH_THRESHOLD:
        return _ALIAS_MAP[match[0]]
    return None


def extract_teams_from_title(title: str) -> list:
    cleaned = re.sub(r"[^\w\s@]", " ", title).strip()
    vs_match = re.search(r"(.+?)\s+(?:vs?\.?\s+|@\s*)(.+?)(?:\s*[-|]|\s*$)", cleaned, re.I)
    if vs_match:
        t1 = _resolve_team(vs_match.group(1).strip())
        t2 = _resolve_team(vs_match.group(2).strip())
        if t1 and t2:
            return [t1, t2]
    found, seen = [], set()
    for alias, (canonical, sport) in _ALIAS_MAP.items():
        if alias in cleaned.lower() and canonical not in seen:
            found.append((canonical, sport))
            seen.add(canonical)
            if len(found) == 2:
                break
    return found


def _detect_sport(teams: list) -> Optional[str]:
    sports = [t[1] for t in teams if t[1]]
    if not sports:
        return None
    return max(set(sports), key=sports.count)


def match_market_to_game(title: str, close_time: str,
                         market_id: str, source: str) -> Optional[dict]:
    from db.historical_store import find_game_by_teams_and_date

    teams = extract_teams_from_title(title)
    if len(teams) < 2:
        return None
    sport = _detect_sport(teams)
    if not sport:
        return None
    close_dt = _parse_ts(close_time)
    if not close_dt:
        return None
    team_names = [t[0] for t in teams]

    for delta in [0, 1, -1, 2, -2]:
        search_date = (close_dt + timedelta(days=delta)).strftime("%Y-%m-%d")
        game = find_game_by_teams_and_date(sport, team_names[0], team_names[1], search_date)
        if not game:
            continue
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        home_s = max(fuzz.WRatio(team_names[0], home),
                     fuzz.WRatio(team_names[1], home)) / 100.0
        away_s = max(fuzz.WRatio(team_names[0], away),
                     fuzz.WRatio(team_names[1], away)) / 100.0
        confidence = round((home_s + away_s) / 2.0 * (1.0 - abs(delta) * 0.04), 3)
        if confidence >= MATCH_THRESHOLD / 100.0:
            return {
                "espn_game_id": game.get("game_id", ""),
                "sport": sport,
                "game_date": search_date,
                "home_team": home,
                "away_team": away,
                "game_start_time": game.get("game_start_time") or f"{search_date}T19:00:00Z",
                "match_confidence_score": confidence,
            }
    return None
