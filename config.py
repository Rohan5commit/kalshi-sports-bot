# config.py

# Trading thresholds
MIN_EDGE = 0.04
ABSTENTION_BAND = (0.40, 0.60)
KELLY_FRACTION = 0.5
MAX_BET_USD = 50.0
DAILY_TRADE_TARGET = (3, 7)

# Auto-relax
AUTO_RELAX_INCREMENT = 0.01
AUTO_RELAX_CAP = 0.08
DRY_DAY_THRESHOLD = 2

# Model performance
MIN_ACCURACY_14D = 0.52
ROLLING_ACCURACY_WINDOW = 14

SPORTS = ["NBA", "NFL", "MLB"]
HISTORICAL_SEASONS = 4
MAX_RETRIES = 3
BACKOFF_BASE = 2

# Modal
MODEL_VOLUME_NAME = "kalshi-bot-models"
SECRET_KALSHI = "kalshi-secret"
SECRET_SUPABASE = "supabase-secret"
SECRET_SMTP = "smtp-secret"
SECRET_API_SPORTS = "api-sports-secret"
SECRET_NEWS_API = "newsapi-secret"
SECRET_KAGGLE = "kaggle-secret"

# Email
REPORT_FROM_EMAIL = "rohan.santhoshkumar1@gmail.com"
REPORT_TO_EMAIL = "rohan.santhoshkumar1@gmail.com"
SMTP_DEFAULT_HOST = "smtp.gmail.com"
SMTP_DEFAULT_PORT = 587

# Kalshi
KALSHI_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_PROD_BASE = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_USE_DEMO = False

# ESPN
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
ESPN_SPORT_PATHS = {
    "NBA": "basketball/nba",
    "NFL": "football/nfl",
    "MLB": "baseball/mlb",
}

# API-Sports
API_SPORTS_BASE = "https://v1.american-football.api-sports.io"

# Currents API
CURRENTS_API_BASE = "https://api.currentsapi.services/v1"

# Rotowire RSS (no key required)
ROTOWIRE_RSS_BASE = "https://www.rotowire.com/rss/news.php"

# Open-Meteo (no key required)
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Historical data — full history ranges for initial load
# NBA: 2022-23 season through 2025-26 (recent seasons for Kalshi overlap)
NBA_SEASONS = list(range(2022, 2026))
# NFL: 1999 season through 2024
NFL_SEASONS = list(range(1999, 2026))
# MLB: 2019 through 2023 (5 seasons of Statcast — 9 seasons took ~5h, over budget)
MLB_SEASONS = list(range(2019, 2027))

SPORT_SR_TYPE = {"NBA": "nba", "NFL": "nfl", "MLB": "mlb"}
