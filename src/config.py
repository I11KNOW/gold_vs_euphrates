"""Central configuration, constants, and path definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

EUPHRATES_FALLBACK_PATH = DATA_DIR / "euphrates_data.csv"
EUPHRATES_LEGACY_PATH = PROJECT_ROOT / "euphrates_data.csv"
GOLD_FALLBACK_PATH = DATA_DIR / "gold_prices_fallback.csv"

# Data source metadata
GOLD_TICKER = "GC=F"
GOLD_START_DATE = "2016-05-01"

DAHITI_STATION_ID = "41518"
DAHITI_STATION_NAME = "Euphrates, River (Iraq)"
DAHITI_STATION_LAT = 31.5545
DAHITI_STATION_LON = 44.6587
DAHITI_SATELLITE_MISSION = "Sentinel-3A (Pass 0212*)"

DAHITI_API_V2_URL = "https://dahiti.dgfi.tum.de/api/v2/download-water-level/"
DAHITI_STATION_PAGE_URL = f"https://dahiti.dgfi.tum.de/en/{DAHITI_STATION_ID}/water-level-altimetry/"
YAHOO_CHART_V8_URL = f"https://query1.finance.yahoo.com/v8/finance/chart/{GOLD_TICKER}"

# Cache & network parameters
GOLD_CACHE_TTL_SECONDS = 6 * 3600      # 6 hours for daily futures
EUPHRATES_CACHE_TTL_SECONDS = 24 * 3600  # 24 hours for ~27-day satellite altimetry
HTTP_TIMEOUT_SECONDS = 12
PASS_MATCH_TOLERANCE_DAYS = 5

# Alignment mode labels
ALIGNMENT_MONTHLY = "Monthly Mean (Calendar Grid)"
ALIGNMENT_PASS_DATE = "Satellite Pass Date (Nearest Trading Day)"
ALIGNMENT_MODES = (ALIGNMENT_MONTHLY, ALIGNMENT_PASS_DATE)

# Default rolling correlation window (periods)
ROLLING_CORR_WINDOW = 12
