# config.py — all thresholds, constants, and settings for the Kalshi sports bot

# Trading thresholds
MIN_EDGE = 0.04            # minimum probability edge to place a bet (4%)
ABSTENTION_BAND = (0.40, 0.60)  # skip if model output falls in this range
KELLY_FRACTION = 0.5       # half-Kelly sizing
MAX_BET_USD = 50.0         # cap per trade in USD
DAILY_TRADE_TARGET = (3, 7)  # soft target trades per day

# Auto-relax parameters
AUTO_RELAX_INCREMENT = 0.01   # relax MIN_EDGE by 1% per dry day
AUTO_RELAX_CAP = 0.08         # never relax beyond 8%
DRY_DAY_THRESHOLD = 2         # days with no trades before auto-relax kicks in

# Model performance threshold
MIN_ACCURACY_14D = 0.52       # trigger full retrain below 52% rolling accuracy
ROLLING_ACCURACY_WINDOW = 14  # days for rolling accuracy computation

# Supported sports
SPORTS = ["NBA", "NFL", "MLB"]

# Historical data
HISTORICAL_SEASONS = 4        # number of seasons to pull for initial training

# API retry settings
MAX_RETRIES = 3
BACKOFF_BASE = 2              # exponential backoff base (seconds)

# Modal Volume name
MODEL_VOLUME_NAME = "kalshi-bot-models"

# Modal Secret names
SECRET_KALSHI = "kalshi-secret"
SECRET_SUPABASE = "supabase-secret"
SECRET_SENDGRID = "sendgrid-secret"
SECRET_API_SPORTS = "api-sports-secret"
SECRET_ODDS_API = "odds-api-secret"
SECRET_NEWS_API = "newsapi-secret"

# Email report settings
REPORT_FROM_EMAIL = "bot@kalshisportsbot.com"
REPORT_TO_EMAIL = "rohan.santhoshkumar1@gmail.com"

# Kalshi API base URLs
KALSHI_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_PROD_BASE = "https://api.kalshi.co/trade-api/v2"
KALSHI_USE_DEMO = True        # always use demo account

# ESPN API base
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
ESPN_SPORT_PATHS = {
    "NBA": "basketball/nba",
    "NFL": "football/nfl",
    "MLB": "baseball/mlb",
}

# API-Sports base
API_SPORTS_BASE = "https://v1.american-football.api-sports.io"  # varies per sport

# The Odds API
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_SPORTS = {
    "NBA": "basketball_nba",
    "NFL": "americanfootball_nfl",
    "MLB": "baseball_mlb",
}

# NewsAPI
NEWS_API_BASE = "https://newsapi.org/v2"

# Sportsreference seasons map
SPORT_SR_TYPE = {
    "NBA": "nba",
    "NFL": "nfl",
    "MLB": "mlb",
}
