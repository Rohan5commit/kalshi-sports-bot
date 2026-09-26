# config.py

# Trading thresholds
MIN_EDGE = 0.01           # 1% minimum edge — auto-relaxes to 0.5% after dry days
KELLY_FRACTION = 0.25     # quarter-Kelly — conservative sizing to limit drawdown
MAX_BET_USD = 25.0        # cap single-game exposure
DAILY_TRADE_TARGET = (2, 5)

# Auto-relax — kicks in after 1 dry day
AUTO_RELAX_INCREMENT = 0.005
AUTO_RELAX_CAP = 0.005
DRY_DAY_THRESHOLD = 1

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

# P&L baseline
BASELINE_BANKROLL = 304.01
BASELINE_RESET_DATE = "2026-07-07"

# Email
REPORT_FROM_EMAIL = "rohan.santhoshkumar1@gmail.com"
REPORT_TO_EMAIL = "rohan.santhoshkumar1@gmail.com"
SMTP_DEFAULT_HOST = "smtp.gmail.com"
SMTP_DEFAULT_PORT = 587

# Kalshi — production only
KALSHI_PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"

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

# Historical data
NBA_SEASONS = list(range(2022, 2026))
NFL_SEASONS = list(range(1999, 2026))
MLB_SEASONS = list(range(2019, 2027))

SPORT_SR_TYPE = {"NBA": "nba", "NFL": "nfl", "MLB": "mlb"}

# Paper trading
PAPER_TRADING = True
PAPER_BANKROLL_START = 500.0
MIN_BET_USD = 5.0
MAX_BET_PCT = 0.05
ORDERBOOK_DEPTH = 20
