# MLBSharp — MLB Strikeout Prop Projection System

A data-driven engine that projects starting pitcher strikeout totals for MLB games. Runs automatically every day during the season via GitHub Actions and serves a live web dashboard.

## What It Does

For each starting pitcher on a given day, the model:

1. Pulls Statcast pitch-by-pitch data (via pybaseball) and FanGraphs leaderboard stats
2. Constructs an arsenal-level matchup score — comparing the pitcher's per-pitch whiff rates against the opposing lineup's vulnerability on those same pitch types
3. Adjusts for park factor, weather (temperature, wind), umpire strikeout tendency, days rest, and pitch count
4. Blends current-season stats with prior-season baselines using a start-count regression schedule to handle small samples early in the year
5. Outputs a projected K total and edge score vs. the book line

## Tech Stack

- **Python** — projection engine, data pipeline, model tuning
- **Flask + Gunicorn** — web dashboard deployed on Render
- **SQLite** — local pitcher database seeded from prior-season FanGraphs stats
- **GitHub Actions** — scheduled daily runs (3x per day), auto-commits updated projections and results
- **pybaseball** — Statcast and FanGraphs data access
- **XGBoost / scikit-learn** — model training once sufficient sample size is reached (200+ logged games)
- **pandas / numpy / scipy** — data wrangling and calibration analysis

## Project Structure

```
run.py                  # daily runner — orchestrates the full pipeline
score_matchups.py       # projection engine (arsenal matchup, adjustments)
fetch_pitcher_stats.py  # Statcast + FanGraphs stat fetcher with blending
fetch_lineup_stats.py   # opposing lineup aggregation by pitch type
fetch_starters.py       # probable starters + confirmed lineups from MLB API
fetch_batter_stats.py   # per-batter Statcast whiff/chase rates by pitch type
fetch_umpires.py        # umpire K-rate tendency lookup
fetch_weather.py        # OpenWeather API — temp, wind speed/direction
build_pitcher_db.py     # one-time DB seeder — prior season stats for all 30 rosters
tune_model.py           # calibration reports, factor correlations, XGBoost training
log_results.py          # auto-logs actual K results after games complete
app.py                  # Flask dashboard server
config.py               # league constants, park factors, venue coordinates
results.csv             # logged projections + actuals (auto-updated by CI)
```

## How To Run

```bash
pip install -r requirements.txt

# Seed the pitcher database (one-time setup)
python build_pitcher_db.py

# Run projections for today
python run.py

# Run for a specific date
python run.py --date 2026-04-10

# Analyze projection accuracy
python tune_model.py
```

Set `OPENWEATHER_API_KEY` in a `.env` file for weather data.

## Model Notes

- **Arsenal matchup** is the primary signal when pitch-type data is available. It computes expected K rate per pitch type, scales by lineup vulnerability, and adds a chase bonus for above-average out-of-zone swing rates.
- **Regression schedule**: current-season stats are blended with prior-season baselines, ramping from 0% → 100% current weight over the first 6 starts. Anchors to league average for rookies with no prior data.
- **XGBoost** layer is withheld until 200 logged games to avoid overfitting on a short sample. Manual projection is used until then.
- Results are tracked in `results.csv` and analyzed with `tune_model.py` for ongoing calibration.

## CI / Automation

`.github/workflows/daily_report.yml` runs the full pipeline at 11 AM, 2:30 PM, and 5 PM CT. It commits updated `dashboard_data.json`, `projections_cache.json`, `fg_cache.json`, and `results.csv` back to the repo so the live dashboard always reflects the latest data.
