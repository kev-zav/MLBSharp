import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# --- MLB Stats API (no key needed) ---
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# --- The Odds API ---
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "baseball_mlb"
ODDS_MARKETS = "pitcher_strikeouts"
ODDS_REGIONS = "us"
ODDS_FORMATS = "american"

# --- Season / Year ---
from datetime import date
SEASON = date.today().year

# --- Projection Settings ---
LEAGUE_AVG_K_PCT   = 0.224        # ~22.4% league average K%
LEAGUE_AVG_BB_PCT  = 0.082        # ~8.2% league average BB%
LEAGUE_AVG_SWSTR   = 0.115        # ~11.5% league average SwStr%
LEAGUE_AVG_CSW     = 0.270        # ~27.0% league average CSW%
LEAGUE_AVG_IP_PER_START     = 4.9
LEAGUE_AVG_PITCHES_PER_START = 88.0
DEFAULT_BATTERS_FACED = 19        # empirically optimal from 95-game sample (Apr 2026)
DEFAULT_PITCH_LIMIT = 85          # league average pitches/start

# --- Opener / Bulk Reliever Exclusion ---
# Pitchers excluded from projections because they are openers or bulk relievers,
# not true starters. Add by MLB pitcher ID where possible (more reliable than name).
EXCLUDED_PITCHER_IDS = {
    691799,  # Grant Taylor (CWS opener — 1.0 IP/start in DB)
}
EXCLUDED_PITCHER_NAMES = {
    "Grant Taylor",
}
# Minimum avg IP/start to be considered a starter. Below this = opener/reliever.
MIN_IP_PER_START = 3.0

# --- Edge Thresholds (percentage points) ---
EDGE_STRONG = 8
EDGE_MODERATE = 4
EDGE_LEAN = 1

# --- Hit Rate Model Calibration ---
# Empirical std dev of (actual_ks - projected_ks) from logged results.
# Update this as more games are logged. Current: 129 games as of 2026-04-07.
HIT_RATE_STD_DEV = 2.675
# Systematic bias: model over-projects by this many Ks on average.
# Applied as a correction when estimating hit rate. Positive = model over-projects.
HIT_RATE_BIAS = 0.813

# --- Park Factors (FanGraphs SO park factor, 100 = neutral) ---
# Loaded dynamically but fallback dict for known parks
PARK_FACTOR_DEFAULT = 100

# --- Venue coordinates for weather lookups ---
VENUE_COORDS = {
    "Angel Stadium":          (33.8003, -117.8827),
    "Busch Stadium":          (38.6226, -90.1928),
    "Chase Field":            (33.4455, -112.0667),
    "Citi Field":             (40.7571, -73.8458),
    "Citizens Bank Park":     (39.9061, -75.1665),
    "Comerica Park":          (42.3390, -83.0485),
    "Coors Field":            (39.7559, -104.9942),
    "Dodger Stadium":         (34.0739, -118.2400),
    "Fenway Park":            (42.3467, -71.0972),
    "Globe Life Field":       (32.7473, -97.0845),
    "Great American Ball Park": (39.0974, -84.5082),
    "Guaranteed Rate Field":  (41.8299, -87.6338),
    "Kauffman Stadium":       (39.0517, -94.4803),
    "loanDepot park":         (25.7781, -80.2196),
    "Minute Maid Park":       (29.7573, -95.3555),
    "Nationals Park":         (38.8730, -77.0074),
    "Oakland Coliseum":       (37.7516, -122.2005),
    "Oracle Park":            (37.7786, -122.3893),
    "Oriole Park at Camden Yards": (39.2838, -76.6216),
    "Petco Park":             (32.7076, -117.1570),
    "PNC Park":               (40.4468, -80.0058),
    "Progressive Field":      (41.4962, -81.6852),
    "Rogers Centre":          (43.6414, -79.3894),
    "T-Mobile Park":          (47.5914, -122.3325),
    "Target Field":           (44.9818, -93.2775),
    "Tropicana Field":        (27.7682, -82.6534),
    "Truist Park":            (33.8908, -84.4678),
    "Wrigley Field":          (41.9484, -87.6553),
    "Yankee Stadium":         (40.8296, -73.9262),
}
