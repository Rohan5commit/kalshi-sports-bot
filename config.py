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
KALSHI_PROD_BASE = "https://api.kalshi.co/trade-api/v2"
KALSHI_USE_DEMO = True

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

# Historical data — seasons to pull on initial load
NBA_SEASONS = [2020, 2021, 2022, 2023]
NFL_SEASONS = [2020, 2021, 2022, 2023]
MLB_SEASONS = [2021, 2022, 2023]

SPORT_SR_TYPE = {"NBA": "nba", "NFL": "nfl", "MLB": "mlb"}
