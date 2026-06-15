"""
data/scrapers/weather.py — Open-Meteo weather data (no API key required).
Historical weather via archive API, forecasts via forecast API.
Only called for outdoor NFL/MLB stadiums — NBA arenas always return None.
"""
import time
from typing import Optional

import requests

from config import MAX_RETRIES, BACKOFF_BASE, OPEN_METEO_FORECAST_URL, OPEN_METEO_ARCHIVE_URL

_DAILY_PARAMS = "temperature_2m_max,precipitation_sum,windspeed_10m_max"

# Outdoor stadium coordinates. Domes/fully-retractable listed in DOME_VENUES are skipped.
STADIUM_COORDS: dict = {
    # ── NFL outdoor ───────────────────────────────────────────────────────────
    "BUF": (42.7738, -78.7870),   # Highmark Stadium
    "NE":  (42.0909, -71.2643),   # Gillette Stadium
    "NYJ": (40.8128, -74.0742),   # MetLife (open-air)
    "NYG": (40.8128, -74.0742),   # MetLife (open-air)
    "PIT": (40.4468, -80.0158),   # Acrisure Stadium
    "CLE": (41.5061, -81.6995),   # Cleveland Browns Stadium
    "CIN": (39.0954, -84.5160),   # Paycor Stadium
    "BAL": (39.2780, -76.6227),   # M&T Bank Stadium
    "KC":  (39.0489, -94.4839),   # Arrowhead Stadium
    "DEN": (39.7439, -105.0201),  # Empower Field at Mile High
    "CHI": (41.8623, -87.6167),   # Soldier Field
    "GB":  (44.5013, -88.0622),   # Lambeau Field
    "SEA": (47.5952, -122.3316),  # Lumen Field
    "SF":  (37.4032, -121.9698),  # Levi's Stadium
    "LAR": (33.9534, -118.3390),  # SoFi Stadium (open-air)
    "LAC": (33.9534, -118.3390),  # SoFi Stadium (open-air)
    "TEN": (36.1665, -86.7713),   # Nissan Stadium
    "JAX": (30.3239, -81.6374),   # EverBank Stadium
    "MIA": (25.9580, -80.2389),   # Hard Rock Stadium
    "TB":  (27.9759, -82.5033),   # Raymond James Stadium
    "CAR": (35.2258, -80.8530),   # Bank of America Stadium
    "PHI": (39.9012, -75.1676),   # Lincoln Financial Field
    "WAS": (38.9076, -76.8644),   # Northwest Stadium
    # ── MLB outdoor ───────────────────────────────────────────────────────────
    "ATL":     (33.8908, -84.4678),  # Truist Park
    "BAL_MLB": (39.2839, -76.6217),  # Camden Yards
    "BOS":     (42.3467, -71.0972),  # Fenway Park
    "CHC":     (41.9484, -87.6553),  # Wrigley Field
    "CHW":     (41.8300, -87.6338),  # Guaranteed Rate Field
    "CIN_MLB": (39.0979, -84.5067),  # Great American Ball Park
    "CLE_MLB": (41.4963, -81.6852),  # Progressive Field
    "COL":     (39.7559, -104.9942), # Coors Field
    "DET":     (42.3390, -83.0485),  # Comerica Park
    "KC_MLB":  (39.0514, -94.4803),  # Kauffman Stadium
    "LAD":     (34.0739, -118.2400), # Dodger Stadium
    "MIA_MLB": (25.7781, -80.2197),  # loanDepot Park
    "MIN_MLB": (44.9817, -93.2783),  # Target Field
    "NYM":     (40.7571, -73.8458),  # Citi Field
    "NYY":     (40.8296, -73.9262),  # Yankee Stadium
    "OAK_MLB": (37.7516, -122.2005), # Oakland Coliseum
    "PHI_MLB": (39.9057, -75.1665),  # Citizens Bank Park
    "PIT_MLB": (40.4468, -80.0057),  # PNC Park
    "SD":      (32.7073, -117.1566), # Petco Park
    "SF_MLB":  (37.7786, -122.3893), # Oracle Park
    "STL":     (38.6226, -90.1928),  # Busch Stadium
    "WSN":     (38.8731, -77.0074),  # Nationals Park
    "LAA":     (33.8003, -117.8827), # Angel Stadium
}

# Venues that are fully enclosed domes or climate-controlled — skip weather API call
DOME_VENUES = {
    "NO", "DAL", "IND", "DET_NFL", "MIN_NFL", "HOU_NFL",   # NFL domes
    "ATL_NFL", "LV",                                        # NFL domes
    "HOU", "MIL", "ARI", "TB_MLB", "TEX", "TOR", "SEA_MLB", # MLB retractable/domes
}

# NFL teams that always play in domes (at home)
_NFL_DOME_TEAMS = {"NO", "DAL", "IND", "MIN", "HOU", "DET", "ATL", "LV"}

# MLB teams in fully enclosed venues
_MLB_DOME_TEAMS = {"TB", "TEX", "TOR", "ARI", "MIL", "SEA", "HOU"}


def _weather_key(sport: str, team: str) -> Optional[str]:
    """Map team abbreviation → STADIUM_COORDS key, or None if indoor/dome."""
    if sport == "NBA":
        return None
    if sport == "NFL":
        if team in _NFL_DOME_TEAMS:
            return None
        return team if team in STADIUM_COORDS else None
    if sport == "MLB":
        if team in _MLB_DOME_TEAMS:
            return None
        mlb_key = f"{team}_MLB"
        if mlb_key in STADIUM_COORDS:
            return mlb_key
        return team if team in STADIUM_COORDS else None
    return None


def _open_meteo(url: str, lat: float, lon: float, date_str: str) -> Optional[dict]:
    """Call Open-Meteo for a single date. Returns {temp_max, precip_sum, windspeed_max} or None."""
    if not isinstance(lat, float) or not isinstance(lon, float):
        print(f"Invalid coordinates: lat={lat}, lon={lon}, skipping weather")
        return None
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": _DAILY_PARAMS,
        "timezone": "America/New_York",
        "start_date": date_str,
        "end_date": date_str,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            return {
                "temp_max": (daily.get("temperature_2m_max") or [None])[0],
                "precip_sum": (daily.get("precipitation_sum") or [None])[0],
                "windspeed_max": (daily.get("windspeed_10m_max") or [None])[0],
            }
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
            else:
                print(f"Open-Meteo error ({url}, {date_str}): {exc}")
    return None


def get_historical_weather(lat: float, lon: float, date_str: str) -> Optional[dict]:
    return _open_meteo(OPEN_METEO_ARCHIVE_URL, lat, lon, date_str)


def get_forecast_weather(lat: float, lon: float, date_str: str) -> Optional[dict]:
    return _open_meteo(OPEN_METEO_FORECAST_URL, lat, lon, date_str)


def get_weather_for_game(sport: str, home_team: str, game_date: str,
                         is_future: bool = False) -> Optional[dict]:
    """
    Main interface. Returns {temp_max, precip_sum, windspeed_max} or None.
    None means indoor/dome or coordinates unknown — caller treats as neutral.
    """
    key = _weather_key(sport, home_team)
    if not key:
        return None
    coords = STADIUM_COORDS.get(key)
    if not coords:
        return None
    lat, lon = coords
    if not isinstance(lat, float) or not isinstance(lon, float):
        print(f"Invalid coordinates for {home_team}: lat={lat}, lon={lon}, skipping weather")
        return None
    url = OPEN_METEO_FORECAST_URL if is_future else OPEN_METEO_ARCHIVE_URL
    return _open_meteo(url, lat, lon, game_date)


def compute_weather_impact_score(weather: Optional[dict]) -> float:
    """
    Single normalised score [0, 1] — higher = more adverse.
    0 = perfect conditions; 1 = extreme cold/wind/rain.
    Open-Meteo returns temperature in Celsius, wind in km/h, precip in mm.
    """
    if not weather:
        return 0.0
    temp_c = float(weather.get("temp_max") or 21.0)
    wind_kmh = float(weather.get("windspeed_max") or 0.0)
    precip_mm = float(weather.get("precip_sum") or 0.0)

    # Cold penalty: 0 at 21°C, 1 at -19°C
    temp_impact = max(0.0, min(1.0, (21.0 - temp_c) / 40.0))
    # Wind penalty: 0 at calm, 1 at 60 km/h (~37 mph)
    wind_impact = min(1.0, wind_kmh / 60.0)
    # Precip penalty: 0 at dry, 1 at 20 mm
    precip_impact = min(1.0, precip_mm / 20.0)

    return float((temp_impact + wind_impact + precip_impact) / 3.0)
