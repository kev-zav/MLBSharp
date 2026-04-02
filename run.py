#!/usr/bin/env python3
"""
MLB Strikeout Sharp — Daily Runner
Usage: python run.py [--date YYYY-MM-DD]
"""

import os
import sys
import time
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from config import PARK_FACTOR_DEFAULT


def main():
    # Parse optional date argument
    game_date = None
    if "--date" in sys.argv:
        idx = sys.argv.index("--date")
        if idx + 1 < len(sys.argv):
            game_date = sys.argv[idx + 1]

    print(f"\n{'='*50}")
    print(f"  MLB STRIKEOUT SHARP — Loading data...")
    print(f"{'='*50}\n")

    # Step 1: Fetch odds FIRST to determine the target date
    print("[1/6] Fetching strikeout odds...")
    t0 = time.time()
    from fetch_odds import fetch_strikeout_odds, get_pitcher_odds
    all_props, odds_date = fetch_strikeout_odds()
    print(f"  ({time.time()-t0:.1f}s)")

    # Use odds date if no explicit date given (odds target the next game day)
    if game_date is None and odds_date:
        game_date = odds_date
        print(f"  → Using odds date: {game_date}")
    elif game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    print(f"\n  Target date: {game_date}\n")

    # Step 2: Fetch probable starters for that date
    print("[2/6] Fetching probable starters...")
    t0 = time.time()
    from fetch_starters import fetch_starters
    matchups = fetch_starters(game_date)
    print(f"  Found {len(matchups)} games with probable starters. ({time.time()-t0:.1f}s)")

    if not matchups:
        print("\n  No games with probable starters found.")
        print("  This could mean:")
        print("    - No games scheduled")
        print("    - Probable pitchers not yet announced")
        print("    - Off-season")
        print("\n  Try: python run.py --date 2025-07-10")
        return

    # Step 3: Fetch weather for all venues
    print("\n[3/6] Fetching weather data...")
    t0 = time.time()
    from fetch_weather import fetch_weather
    weather_cache = {}
    venues = set(m["venue"] for m in matchups)
    for venue in venues:
        weather_cache[venue] = fetch_weather(venue)
    print(f"  Weather data for {len(venues)} venues. ({time.time()-t0:.1f}s)")

    # Step 4: Fetch umpire assignments
    print("\n[4/6] Fetching umpire assignments...")
    t0 = time.time()
    from fetch_umpires import fetch_umpire_for_game
    umpire_cache = {}
    for m in matchups:
        key = m["game_id"]
        umpire_cache[key] = fetch_umpire_for_game(
            m["home_team"]["team_abbr"],
            m["away_team"]["team_abbr"],
            game_date,
        )
    print(f"  ({time.time()-t0:.1f}s)")

    # Step 5: Fetch pitcher stats and lineup stats, then score
    print("\n[5/6] Fetching pitcher & lineup stats and scoring matchups...")
    t0 = time.time()
    from fetch_pitcher_stats import fetch_pitcher_stats
    from fetch_lineup_stats import fetch_lineup_stats
    from score_matchups import score_matchup

    scored = []

    for m in matchups:
        weather = weather_cache.get(m["venue"], {"adjustment": 1.0})
        umpire = umpire_cache.get(m["game_id"], {"adjustment": 1.0})

        # Score both pitchers in each game
        for side in ["away", "home"]:
            pitcher_info = m[f"{side}_pitcher"]
            opp_side = "home" if side == "away" else "away"
            opp_team = m[f"{opp_side}_team"]

            pitcher_name = pitcher_info["name"]
            pitcher_id = pitcher_info["id"]
            pitcher_team_abbr = pitcher_info["team_abbr"]
            opp_abbr = opp_team["team_abbr"]

            print(f"  Scoring: {pitcher_name} ({pitcher_team_abbr}) vs {opp_abbr}...", end="", flush=True)

            try:
                p_stats = fetch_pitcher_stats(pitcher_id, pitcher_name)
            except Exception as e:
                print(f" [pitcher stats error: {e}]")
                continue

            try:
                l_stats = fetch_lineup_stats(opp_abbr, "R")
            except Exception as e:
                print(f" [lineup stats error: {e}]")
                continue

            # Match odds by pitcher name AND team
            odds = get_pitcher_odds(all_props, pitcher_name, pitcher_team_abbr)

            try:
                result = score_matchup(
                    p_stats, l_stats, odds,
                    park_factor=PARK_FACTOR_DEFAULT,
                    weather=weather,
                    umpire=umpire,
                )
            except Exception as e:
                print(f" [scoring error: {e}]")
                continue

            # Enrich result with display info
            result["pitcher_name"] = pitcher_name
            result["pitcher_id"] = pitcher_id
            result["pitcher_team"] = pitcher_team_abbr
            result["opp_team"] = opp_abbr
            result["venue"] = m["venue"]
            result["game_id"] = m["game_id"]
            result["game_time"] = m["game_time"]
            result["swstr_pct"] = p_stats.get("swstr_pct", 0)
            result["csw_pct"] = p_stats.get("csw_pct", 0)
            result["k_pct"] = p_stats.get("k_pct", 0)
            result["pitcher_hand"] = "R"
            result["opp_k_pct"] = l_stats.get("team_k_pct", 0)
            result["opp_chase"] = l_stats.get("o_swing_pct", 0)
            result["park_factor"] = PARK_FACTOR_DEFAULT
            result["umpire"] = umpire
            result["weather"] = weather
            result["days_rest"] = p_stats.get("days_rest", 5)
            result["last_outing_pitches"] = p_stats.get("last_outing_pitches", 0)
            result["rolling_k_3"] = p_stats.get("rolling_k_3", 0)
            result["rolling_k_5"] = p_stats.get("rolling_k_5", 0)

            scored.append(result)
            edge_str = f"edge {result['edge']:+.1f}" if result['edge'] != 0 else "no odds"
            print(f" → {result['projected_ks']} Ks proj, {edge_str}")

    print(f"\n  Scored {len(scored)} pitcher matchups. ({time.time()-t0:.1f}s)")

    # Step 6: Generate report
    print("\n[6/6] Generating report...\n")
    from report import generate_report
    report = generate_report(scored)
    print(report)

    # Save report to file
    filename = f"report_{game_date}.txt"
    with open(filename, "w") as f:
        f.write(report)
    print(f"\nReport saved to {filename}")

    # Save projections cache for log_results.py
    import json
    cache_data = []
    for s in scored:
        cache_data.append({
            "date": game_date,
            "pitcher_name": s.get("pitcher_name", ""),
            "pitcher_id": s.get("pitcher_id", 0),
            "pitcher_hand": s.get("pitcher_hand", "R"),
            "pitcher_team": s.get("pitcher_team", ""),
            "opp_team": s.get("opp_team", ""),
            "venue": s.get("venue", ""),
            "game_id": s.get("game_id", ""),
            "projected_ks": s.get("projected_ks", 0),
            "line": s.get("line", 0),
            "edge": s.get("edge", 0),
            "play": s.get("play", ""),
            "hit_rate": s.get("hit_rate", 0),
            "swstr_pct": s.get("swstr_pct", 0),
            "csw_pct": s.get("csw_pct", 0),
            "k_pct": s.get("k_pct", 0),
            "opp_k_pct": s.get("opp_k_pct", 0),
            "opp_chase": s.get("opp_chase", 0),
            "park_factor": s.get("park_factor", 100),
            "ump_adjustment": s.get("umpire", {}).get("adjustment", 1.0),
            "ump_name": s.get("umpire", {}).get("name", "Unknown"),
            "weather_temp": s.get("weather", {}).get("temp_f", 72),
            "weather_wind": s.get("weather", {}).get("wind_mph", 0),
            "days_rest": s.get("days_rest", 5),
            "last_outing_pitches": s.get("last_outing_pitches", 0),
            "rolling_k_3": s.get("rolling_k_3", 0),
            "rolling_k_5": s.get("rolling_k_5", 0),
        })

    cache_path = os.path.join(os.path.dirname(__file__) or ".", "projections_cache.json")
    # Load existing cache and append/update for this date
    existing = {}
    try:
        with open(cache_path) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    existing[game_date] = cache_data
    with open(cache_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Projections cached to projections_cache.json ({len(cache_data)} entries)")

    # Save dashboard data for the web app
    from config import EDGE_STRONG, EDGE_MODERATE, EDGE_LEAN
    from datetime import datetime

    def make_play(s):
        return {
            "pitcher_name": s.get("pitcher_name", ""),
            "pitcher_team": s.get("pitcher_team", ""),
            "opp_team": s.get("opp_team", ""),
            "venue": s.get("venue", ""),
            "projected_ks": s.get("projected_ks", 0),
            "line": s.get("line", 0),
            "play": s.get("play", ""),
            "edge": s.get("edge", 0),
            "hit_rate": s.get("hit_rate", 0),
            "swstr_pct": s.get("swstr_pct", 0),
            "csw_pct": s.get("csw_pct", 0),
            "k_pct": s.get("k_pct", 0),
            "opp_k_pct": s.get("opp_k_pct", 0),
            "opp_chase": s.get("opp_chase", 0),
            "days_rest": s.get("days_rest", 5),
            "rolling_k_3": s.get("rolling_k_3", 0),
            "rolling_k_5": s.get("rolling_k_5", 0),
            "umpire": s.get("umpire", {}),
            "weather": s.get("weather", {}),
            "over_odds": s.get("over_odds", []),
            "under_odds": s.get("under_odds", []),
            "best_line": s.get("best_line"),
            "k_distribution": {
                str(k): v for k, v in s.get("k_distribution", {}).items()
            },
        }

    sorted_scored = sorted(scored, key=lambda x: x.get("edge", 0), reverse=True)
    strong = [make_play(s) for s in sorted_scored if s.get("edge", 0) >= EDGE_STRONG]
    moderate = [make_play(s) for s in sorted_scored if EDGE_MODERATE <= s.get("edge", 0) < EDGE_STRONG]
    lean = [make_play(s) for s in sorted_scored if EDGE_LEAN <= s.get("edge", 0) < EDGE_MODERATE]
    no_value = [make_play(s) for s in sorted_scored if s.get("edge", 0) < EDGE_LEAN]

    # Build parlay suggestions
    candidates = strong + moderate
    parlays = []
    if len(candidates) >= 2:
        t = candidates
        legs2 = f"{t[0]['pitcher_name']} {t[0]['play']} {t[0]['line']} + {t[1]['pitcher_name']} {t[1]['play']} {t[1]['line']}"
        books2 = ", ".join(filter(None, [
            t[0].get("best_line", {}).get("book") if t[0].get("best_line") else None,
            t[1].get("best_line", {}).get("book") if t[1].get("best_line") else None,
        ]))
        parlays.append({"legs": legs2, "books": books2})
    if len(candidates) >= 3:
        t = candidates
        legs3 = " + ".join(f"{p['pitcher_name']} {p['play']} {p['line']}" for p in t[:3])
        parlays.append({"legs": legs3, "books": ""})

    dashboard = {
        "date": datetime.strptime(game_date, "%Y-%m-%d").strftime("%B %d, %Y"),
        "generated_at": datetime.now().strftime("%I:%M %p"),
        "strong": strong,
        "moderate": moderate,
        "lean": lean,
        "no_value": no_value,
        "parlays": parlays,
        "total_plays": len(scored),
    }

    dashboard_path = os.path.join(os.path.dirname(__file__) or ".", "dashboard_data.json")
    with open(dashboard_path, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"Dashboard data saved to dashboard_data.json")

    # Push dashboard data to GitHub so Render serves the latest report
    import subprocess
    print("\nPushing dashboard to GitHub...")
    try:
        subprocess.run(["git", "add", "dashboard_data.json"], cwd=SCRIPT_DIR, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Dashboard update {game_date}"],
            cwd=SCRIPT_DIR, check=True, capture_output=True
        )
        subprocess.run(["git", "push"], cwd=SCRIPT_DIR, check=True, capture_output=True)
        print("Dashboard live at https://mlbsharp.onrender.com")
    except subprocess.CalledProcessError:
        print("Git push skipped (no changes or auth issue)")


if __name__ == "__main__":
    main()
