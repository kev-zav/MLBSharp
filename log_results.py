#!/usr/bin/env python3
"""
MLB Strikeout Sharp — Results Logger
Pulls actual K totals from MLB Stats API box scores, matches to projections,
and appends to results.csv.

Usage:
  python3 log_results.py                    # logs yesterday's results
  python3 log_results.py --date 2026-04-01  # logs a specific date
"""

import csv
import json
import os
import sys
from datetime import date, timedelta

import requests
from config import MLB_API_BASE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")
PROJECTIONS_CACHE = os.path.join(SCRIPT_DIR, "projections_cache.json")

CSV_COLUMNS = [
    "date",
    "pitcher_name",
    "pitcher_hand",
    "team",
    "opponent",
    "venue",
    "projected_ks",
    "actual_ks",
    "swstr_pct",
    "csw_pct",
    "k_pct",
    "opp_k_pct_vs_hand",
    "o_swing_pct",
    "park_factor",
    "ump_adjustment",
    "weather_temp",
    "weather_wind",
    "days_rest",
    "pitch_count_last_outing",
    "rolling_k_avg_3",
    "rolling_k_avg_5",
    "pitch_mix_adj",
    "projected_ks_manual",
    "innings_pitched",
]


def get_actual_ks(game_date: str) -> list[dict]:
    """
    Pull actual pitcher strikeout totals from MLB Stats API box scores.
    Fetches the schedule first, then each game's boxscore individually.
    Returns list of {game_id, pitcher_name, pitcher_id, team_abbr, opp_abbr,
                     venue, actual_ks, actual_pitches, innings_pitched}.
    """
    # Get schedule for the date
    url = f"{MLB_API_BASE}/schedule"
    params = {"sportId": 1, "date": game_date, "hydrate": "venue"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results = []

    for d in data.get("dates", []):
        for game in d.get("games", []):
            status = game.get("status", {}).get("detailedState", "")
            if status not in ("Final", "Completed Early"):
                continue

            game_id = game.get("gamePk")
            venue = game.get("venue", {}).get("name", "Unknown")

            # Fetch boxscore for this game
            try:
                box_resp = requests.get(
                    f"{MLB_API_BASE}/game/{game_id}/boxscore", timeout=15
                )
                box_resp.raise_for_status()
                boxscore = box_resp.json()
            except Exception:
                continue

            teams_box = boxscore.get("teams", {})

            for side in ["away", "home"]:
                opp_side = "home" if side == "away" else "away"
                team_data = teams_box.get(side, {})
                team_info = team_data.get("team", {})
                team_abbr = team_info.get("abbreviation", "")

                opp_data = teams_box.get(opp_side, {})
                opp_info = opp_data.get("team", {})
                opp_abbr = opp_info.get("abbreviation", "")

                players = team_data.get("players", {})
                pitcher_order = team_data.get("pitchers", [])

                # First pitcher in the order is the starter
                if pitcher_order:
                    starter_id = pitcher_order[0]
                    pdata = players.get(f"ID{starter_id}", {})
                    if pdata:
                        pstats = pdata.get("stats", {}).get("pitching", {})
                        pitcher_name = pdata.get("person", {}).get("fullName", "Unknown")
                        actual_ks = int(pstats.get("strikeOuts", 0))
                        actual_pitches = int(pstats.get("numberOfPitches", 0))
                        ip = pstats.get("inningsPitched", "0")

                        results.append({
                            "game_id": game_id,
                            "pitcher_name": pitcher_name,
                            "pitcher_id": starter_id,
                            "team_abbr": team_abbr,
                            "opp_abbr": opp_abbr,
                            "venue": venue,
                            "actual_ks": actual_ks,
                            "actual_pitches": actual_pitches,
                            "innings_pitched": ip,
                        })

    return results


def load_projections(game_date: str) -> list[dict]:
    """Load cached projections for a given date."""
    try:
        with open(PROJECTIONS_CACHE) as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    return cache.get(game_date, [])


def match_projection(actual: dict, projections: list[dict]) -> dict | None:
    """Match an actual result to its projection by pitcher name and team."""
    actual_last = actual["pitcher_name"].strip().split()[-1].lower()
    actual_team = actual["team_abbr"]

    for proj in projections:
        proj_last = proj["pitcher_name"].strip().split()[-1].lower()
        proj_team = proj.get("pitcher_team", "")

        if proj_last == actual_last and proj_team == actual_team:
            return proj

        # Fallback: just last name match
        if proj_last == actual_last:
            return proj

    return None


def ensure_csv():
    """Create results.csv with headers if it doesn't exist."""
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
        print(f"  Created {RESULTS_CSV} with headers.")


def check_already_logged(game_date: str) -> set:
    """Return set of pitcher names already logged for this date."""
    logged = set()
    if not os.path.exists(RESULTS_CSV):
        return logged

    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("date") == game_date:
                logged.add(row.get("pitcher_name", ""))
    return logged


def log_results(game_date: str):
    """Main entry point: pull actuals, match to projections, write CSV rows."""
    print(f"\n{'='*50}")
    print(f"  MLB STRIKEOUT SHARP — Results Logger")
    print(f"  Date: {game_date}")
    print(f"{'='*50}\n")

    ensure_csv()

    # Check what's already logged
    already_logged = check_already_logged(game_date)
    if already_logged:
        print(f"  Already logged {len(already_logged)} pitchers for {game_date}.")

    # Pull actual results
    print("[1/3] Fetching actual K totals from box scores...")
    actuals = get_actual_ks(game_date)
    print(f"  Found {len(actuals)} starting pitcher results.")

    if not actuals:
        print("\n  No final box scores found for this date.")
        print("  Games may not have been played yet or data isn't available.")
        return

    # Load projections
    print("\n[2/3] Loading cached projections...")
    projections = load_projections(game_date)
    if projections:
        print(f"  Found {len(projections)} cached projections.")
    else:
        print("  No cached projections found for this date.")
        print("  Results will be logged with blank projection columns.")
        print("  (Run run.py --date YYYY-MM-DD first to cache projections)")

    # Match and write
    print("\n[3/3] Matching results to projections and writing CSV...")
    rows_written = 0
    rows_skipped = 0

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)

        for actual in actuals:
            pitcher_name = actual["pitcher_name"]

            if pitcher_name in already_logged:
                rows_skipped += 1
                continue

            proj = match_projection(actual, projections)

            row = [
                game_date,
                pitcher_name,
                proj.get("pitcher_hand", "R") if proj else "R",
                actual["team_abbr"],
                actual["opp_abbr"],
                actual["venue"],
                proj.get("projected_ks", "") if proj else "",
                actual["actual_ks"],
                proj.get("swstr_pct", "") if proj else "",
                proj.get("csw_pct", "") if proj else "",
                proj.get("k_pct", "") if proj else "",
                proj.get("opp_k_pct", "") if proj else "",
                proj.get("opp_chase", "") if proj else "",
                proj.get("park_factor", "") if proj else "",
                proj.get("ump_adjustment", "") if proj else "",
                proj.get("weather_temp", "") if proj else "",
                proj.get("weather_wind", "") if proj else "",
                proj.get("days_rest", "") if proj else "",
                proj.get("last_outing_pitches", "") if proj else "",
                proj.get("rolling_k_3", "") if proj else "",
                proj.get("rolling_k_5", "") if proj else "",
                proj.get("pitch_mix_adj", "") if proj else "",
                proj.get("projected_ks_manual", "") if proj else "",
                actual["innings_pitched"],
            ]

            writer.writerow(row)
            rows_written += 1

            matched_str = "MATCHED" if proj else "no projection"
            proj_str = f" | proj {proj['projected_ks']}" if proj else ""
            print(f"  {pitcher_name} ({actual['team_abbr']}): "
                  f"{actual['actual_ks']} Ks{proj_str} — [{matched_str}]")

    print(f"\n  Wrote {rows_written} rows to results.csv "
          f"({rows_skipped} skipped as duplicates).")

    # Summary stats
    if projections and rows_written > 0:
        matched = [a for a in actuals if match_projection(a, projections) and a["pitcher_name"] not in already_logged]
        if matched:
            total_proj = sum(match_projection(a, projections)["projected_ks"] for a in matched)
            total_actual = sum(a["actual_ks"] for a in matched)
            avg_proj = total_proj / len(matched)
            avg_actual = total_actual / len(matched)
            print(f"\n  Quick Stats:")
            print(f"    Avg Projected: {avg_proj:.1f} Ks")
            print(f"    Avg Actual:    {avg_actual:.1f} Ks")
            print(f"    Avg Miss:      {avg_proj - avg_actual:+.1f} Ks")


def main():
    game_date = None
    if "--date" in sys.argv:
        idx = sys.argv.index("--date")
        if idx + 1 < len(sys.argv):
            game_date = sys.argv[idx + 1]

    if game_date is None:
        # Default to yesterday (games should be final by morning)
        game_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    log_results(game_date)


if __name__ == "__main__":
    main()
