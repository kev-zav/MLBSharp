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

    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    print(f"\n{'='*50}")
    print(f"  MLB STRIKEOUT SHARP — K Projections")
    print(f"  {game_date}")
    print(f"{'='*50}\n")

    # Step 1: Fetch probable starters
    print("[1/5] Fetching probable starters and lineups...")
    t0 = time.time()
    from fetch_starters import fetch_starters, fetch_lineup_for_game
    matchups = fetch_starters(game_date)
    print(f"  Found {len(matchups)} games with probable starters. ({time.time()-t0:.1f}s)")

    if not matchups:
        print("\n  No games with probable starters found.")
        print("  This could mean:")
        print("    - No games scheduled")
        print("    - Probable pitchers not yet announced")
        print("    - Off-season")
        print("\n  Try: python run.py --date 2026-07-10")
        return

    # Attempt lineup hydration for each game
    lineups_confirmed = 0
    for m in matchups:
        lineup = fetch_lineup_for_game(m["game_id"])
        m["away_lineup"] = lineup["away_batters"] if lineup else None
        m["home_lineup"] = lineup["home_batters"] if lineup else None
        if lineup:
            lineups_confirmed += 1
    print(f"  Lineups confirmed: {lineups_confirmed}/{len(matchups)} games")

    # Step 2: Fetch weather for all venues
    print("\n[2/5] Fetching weather data...")
    t0 = time.time()
    from fetch_weather import fetch_weather
    weather_cache = {}
    venues = set(m["venue"] for m in matchups)
    for venue in venues:
        weather_cache[venue] = fetch_weather(venue)
    print(f"  Weather data for {len(venues)} venues. ({time.time()-t0:.1f}s)")

    # Step 3: Fetch umpire assignments
    print("\n[3/5] Fetching umpire assignments...")
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

    # Step 4: Fetch pitcher stats and lineup stats, then project
    print("\n[4/5] Fetching pitcher & lineup stats and projecting matchups...")
    t0 = time.time()
    from fetch_pitcher_stats import fetch_pitcher_stats
    from fetch_lineup_stats import fetch_lineup_stats
    from score_matchups import project_matchup

    scored = []

    for m in matchups:
        weather = weather_cache.get(m["venue"], {"adjustment": 1.0})
        umpire = umpire_cache.get(m["game_id"], {"adjustment": 1.0})

        for side in ["away", "home"]:
            pitcher_info = m[f"{side}_pitcher"]
            opp_side = "home" if side == "away" else "away"
            opp_team = m[f"{opp_side}_team"]

            pitcher_name = pitcher_info["name"]
            pitcher_id = pitcher_info["id"]
            pitcher_team_abbr = pitcher_info["team_abbr"]
            opp_abbr = opp_team["team_abbr"]
            opp_lineup_ids = m.get(f"{opp_side}_lineup")

            print(f"  Projecting: {pitcher_name} ({pitcher_team_abbr}) vs {opp_abbr}...", end="", flush=True)

            try:
                p_stats = fetch_pitcher_stats(pitcher_id, pitcher_name, pitcher_team_abbr)
            except Exception as e:
                print(f" [pitcher stats error: {e}]")
                continue

            # Skip openers and bulk relievers
            from config import EXCLUDED_PITCHER_IDS, EXCLUDED_PITCHER_NAMES, MIN_IP_PER_START
            if pitcher_id in EXCLUDED_PITCHER_IDS or pitcher_name in EXCLUDED_PITCHER_NAMES:
                print(f" [SKIPPED — known opener/reliever]")
                continue
            ip_per_start = p_stats.get("ip_per_start", 99)
            if ip_per_start > 0 and ip_per_start < MIN_IP_PER_START:
                print(f" [SKIPPED — avg {ip_per_start:.1f} IP/start, likely opener/reliever]")
                continue

            try:
                l_stats = fetch_lineup_stats(
                    opp_abbr,
                    p_stats.get("pitcher_hand", "R"),
                    batter_ids=opp_lineup_ids,
                )
            except Exception as e:
                print(f" [lineup stats error: {e}]")
                continue

            try:
                result = project_matchup(
                    p_stats, l_stats,
                    park_factor=PARK_FACTOR_DEFAULT,
                    weather=weather,
                    umpire=umpire,
                )
            except Exception as e:
                print(f" [projection error: {e}]")
                continue

            result["pitch_mix_adj"] = result.get("pitch_mix_adj", 1.0)
            result["projected_ks_manual"] = result.get("projected_ks", 0)

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
            result["pitcher_hand"] = p_stats.get("pitcher_hand", "R")
            result["opp_k_pct"] = l_stats.get("team_k_pct", 0)
            result["opp_chase"] = l_stats.get("o_swing_pct", 0)
            result["park_factor"] = PARK_FACTOR_DEFAULT
            result["umpire"] = umpire
            result["weather"] = weather
            result["days_rest"] = p_stats.get("days_rest", 5)
            result["last_outing_pitches"] = p_stats.get("last_outing_pitches", 0)
            result["rolling_k_3"] = p_stats.get("rolling_k_3", 0)
            result["rolling_k_5"] = p_stats.get("rolling_k_5", 0)
            result["lineup_confirmed"] = opp_lineup_ids is not None

            # Last 5 game logs for display
            raw_logs = p_stats.get("game_logs", [])
            last5 = []
            for g in raw_logs[-5:]:
                d = g["date"]
                try:
                    d = d.strftime("%m/%d") if hasattr(d, "strftime") else d.date().strftime("%m/%d")
                except Exception:
                    d = str(d)[:10]
                last5.append({"date": d, "k": g["strikeouts"], "pitches": g["pitches"]})
            result["last5"] = last5

            scored.append(result)
            lineup_tag = " [lineup]" if opp_lineup_ids else ""
            print(f" → {result['projected_ks']} Ks{lineup_tag}")

    print(f"\n  Projected {len(scored)} pitcher matchups. ({time.time()-t0:.1f}s)")

    # Deduplicate by pitcher_id — doubleheaders can list the same starter for both games
    seen_pitchers: dict[int, int] = {}
    deduped = []
    for entry in scored:
        pid = entry.get("pitcher_id")
        if pid not in seen_pitchers:
            seen_pitchers[pid] = len(deduped)
            deduped.append(entry)
        else:
            # Keep whichever has higher projected Ks (more complete data)
            existing_idx = seen_pitchers[pid]
            if entry.get("projected_ks", 0) > deduped[existing_idx].get("projected_ks", 0):
                deduped[existing_idx] = entry
                print(f"  [DEDUP] Replaced {entry.get('pitcher_name')} entry")
            else:
                print(f"  [DEDUP] Dropped duplicate {entry.get('pitcher_name')} entry")
    scored = deduped

    # Step 5: Generate report
    print("\n[5/5] Generating report...\n")
    from report import generate_report
    report = generate_report(scored)
    print(report)

    filename = f"report_{game_date}.txt"
    with open(filename, "w") as f:
        f.write(report)
    print(f"\nReport saved to {filename}")

    # Save projections cache
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
            "projected_ks_manual": s.get("projected_ks_manual", s.get("projected_ks", 0)),
            "lineup_confirmed": s.get("lineup_confirmed", False),
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
            "pitch_mix_adj": s.get("pitch_mix_adj", 1.0),
        })

    cache_path = os.path.join(os.path.dirname(__file__) or ".", "projections_cache.json")
    existing = {}
    try:
        with open(cache_path) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Merge into existing cache for this date — don't overwrite pitchers from earlier runs
    existing_today = {e["pitcher_id"]: e for e in existing.get(game_date, [])}
    for entry in cache_data:
        existing_today[entry["pitcher_id"]] = entry
    existing[game_date] = list(existing_today.values())

    with open(cache_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Projections cached to projections_cache.json ({len(existing[game_date])} entries)")

    # Save dashboard data
    from datetime import datetime

    def make_entry(s):
        return {
            "pitcher_name": s.get("pitcher_name", ""),
            "pitcher_id": s.get("pitcher_id", 0),
            "pitcher_team": s.get("pitcher_team", ""),
            "pitcher_hand": s.get("pitcher_hand", "R"),
            "opp_team": s.get("opp_team", ""),
            "venue": s.get("venue", ""),
            "projected_ks": s.get("projected_ks", 0),
            "lineup_confirmed": s.get("lineup_confirmed", False),
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
            "last5": s.get("last5", []),
            "k_distribution": {
                str(k): v for k, v in s.get("k_distribution", {}).items()
            },
        }

    all_pitchers = [make_entry(s) for s in sorted(scored, key=lambda x: x.get("projected_ks", 0), reverse=True)]

    dashboard = {
        "date": datetime.strptime(game_date, "%Y-%m-%d").strftime("%B %d, %Y"),
        "generated_at": datetime.now().strftime("%I:%M %p"),
        "all_pitchers": all_pitchers,
        "total_pitchers": len(scored),
    }

    dashboard_path = os.path.join(os.path.dirname(__file__) or ".", "dashboard_data.json")
    with open(dashboard_path, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"Dashboard data saved to dashboard_data.json")

    # Push to GitHub
    import subprocess
    print("\nPushing dashboard to GitHub...")
    try:
        subprocess.run(["git", "add", "dashboard_data.json", "projections_cache.json", "fg_cache.json"], cwd=SCRIPT_DIR, check=True)
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
