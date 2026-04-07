#!/usr/bin/env python3
"""
MLB Strikeout Sharp — Daily Runner
Usage: python run.py [--date YYYY-MM-DD]
"""

import os
import sys
import time
import pickle
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pkl")

from config import PARK_FACTOR_DEFAULT

# XGBoost feature order must match what tune_model.py trained on
_XGB_FEATURES = [
    "swstr_pct", "csw_pct", "k_pct", "opp_k_pct_vs_hand", "o_swing_pct",
    "park_factor", "ump_adjustment", "weather_temp", "weather_wind",
    "days_rest", "pitch_count_last_outing", "rolling_k_avg_3", "rolling_k_avg_5",
    "projected_ks_manual",   # manual model output as a feature
    "pitch_mix_adj",         # pitch type matchup multiplier
]


def _load_xgb_model():
    """Load XGBoost model if available. Returns (model, features) or (None, None)."""
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        features = data["features"]
        rows = data.get("train_rows", "?")
        mae = data.get("cv_mae", data.get("train_mae", "?"))
        print(f"  [XGB] Loaded model.pkl — trained on {rows} rows, CV MAE: {mae:.2f}")
        return model, features
    except FileNotFoundError:
        return None, None
    except Exception as e:
        print(f"  [XGB] Failed to load model.pkl: {e}")
        return None, None


def _xgb_predict(model, features, p_stats, l_stats, park_factor, weather, umpire, manual_proj: float = 0.0, pitch_mix_adj: float = 1.0):
    """Use XGBoost model to predict Ks. Returns float or None if inputs incomplete."""
    import numpy as np
    row = {
        "swstr_pct": p_stats.get("swstr_pct", 0),
        "csw_pct": p_stats.get("csw_pct", 0),
        "k_pct": p_stats.get("k_pct", 0),
        "opp_k_pct_vs_hand": l_stats.get("team_k_pct", 0),
        "o_swing_pct": l_stats.get("o_swing_pct", 0),
        "park_factor": park_factor,
        "ump_adjustment": umpire.get("adjustment", 1.0),
        "weather_temp": weather.get("temp_f", 72),
        "weather_wind": weather.get("wind_mph", 0),
        "days_rest": p_stats.get("days_rest", 5),
        "pitch_count_last_outing": p_stats.get("last_outing_pitches", 0),
        "rolling_k_avg_3": p_stats.get("rolling_k_3", 0),
        "rolling_k_avg_5": p_stats.get("rolling_k_5", 0),
        "projected_ks_manual": manual_proj,
        "pitch_mix_adj": pitch_mix_adj,
    }
    X = np.array([[row[f] for f in features]])
    pred = float(model.predict(X)[0])
    return round(max(0.5, min(14.0, pred)), 1)


def main():
    # Parse optional date argument
    game_date = None
    if "--date" in sys.argv:
        idx = sys.argv.index("--date")
        if idx + 1 < len(sys.argv):
            game_date = sys.argv[idx + 1]

    no_odds = "--no-odds" in sys.argv

    print(f"\n{'='*50}")
    print(f"  MLB STRIKEOUT SHARP — Loading data...")
    print(f"{'='*50}\n")

    # XGBoost disabled until 200+ graded results are available for reliable training
    xgb_model, xgb_features = None, None

    # Step 1: Fetch odds FIRST to determine the target date (skip with --no-odds)
    from fetch_odds import get_pitcher_odds
    if no_odds:
        print("[1/6] Skipping odds fetch (--no-odds)...")
        all_props = []
        if game_date is None:
            game_date = date.today().strftime("%Y-%m-%d")
    else:
        print("[1/6] Fetching strikeout odds...")
        t0 = time.time()
        from fetch_odds import fetch_strikeout_odds
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
                l_stats = fetch_lineup_stats(opp_abbr, p_stats.get("pitcher_hand", "R"))
            except Exception as e:
                print(f" [lineup stats error: {e}]")
                continue

            # Match odds by pitcher name AND team
            odds = get_pitcher_odds(all_props, pitcher_name, pitcher_team_abbr)

            # Starter change detection — if no odds found, check if a different pitcher
            # from the same team has odds (indicates probable starter may have changed)
            if odds["line"] == 0 and all_props:
                team_props = [
                    p["pitcher_name"] for p in all_props
                    if p.get("home_abbr") == pitcher_team_abbr or p.get("away_abbr") == pitcher_team_abbr
                ]
                team_props = list(set(team_props))
                if team_props:
                    print(f"\n  [STARTER CHANGE?] No odds for {pitcher_name} ({pitcher_team_abbr}) "
                          f"but odds exist for: {', '.join(team_props)} — probable starter may have changed")

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

            result["pitch_mix_adj"] = result.get("pitch_mix_adj", 1.0)

            # Save manual projection before XGB may override it
            result["projected_ks_manual"] = result.get("projected_ks", 0)

            # Stack XGBoost on top of manual model (manual proj is a feature)
            if xgb_model is not None:
                manual_proj = result.get("projected_ks", 0)
                pitch_mix_adj = result.get("pitch_mix_adj", 1.0)
                xgb_proj = _xgb_predict(
                    xgb_model, xgb_features, p_stats, l_stats,
                    PARK_FACTOR_DEFAULT, weather, umpire,
                    manual_proj=manual_proj,
                    pitch_mix_adj=pitch_mix_adj,
                )
                if xgb_proj is not None:
                    # If XGB diverges from manual by more than 2K, blend 50/50
                    # XGB has limited training data early season and can produce outliers
                    deviation = abs(xgb_proj - manual_proj)
                    if deviation >= 1.5:
                        blended = round((xgb_proj + manual_proj) / 2, 1)
                        print(f"\n  [XGB] {pitcher_name}: XGB={xgb_proj} vs manual={manual_proj} "
                              f"(diff {deviation:.1f}K > 2K) → blended to {blended}")
                        result["projected_ks"] = blended
                        result["projection_source"] = "xgboost_blended"
                    else:
                        result["projected_ks"] = xgb_proj
                        result["projection_source"] = "xgboost"
                else:
                    result["projection_source"] = "manual"
            else:
                result["projection_source"] = "manual"

            from score_matchups import calc_hit_rate, calc_edge, calc_fair_value_odds

            # Recalculate hit rate / edge / play using final projected_ks
            # (XGBoost may have overridden the manual model's projection)
            final_proj = result["projected_ks"]
            line_val = odds.get("line", 0)
            best_over_o = odds.get("best_over")
            best_under_o = odds.get("best_under")
            over_hr = calc_hit_rate(final_proj, line_val)
            under_hr = round(100 - over_hr, 1)
            over_edge_r = calc_edge(over_hr, best_over_o["odds"]) if best_over_o else 0.0
            under_edge_r = calc_edge(under_hr, best_under_o["odds"]) if best_under_o else 0.0
            if line_val <= 0 or final_proj >= line_val:
                result["play"] = "OVER"
                result["edge"] = over_edge_r
                result["hit_rate"] = over_hr
                result["best_line"] = best_over_o
            else:
                result["play"] = "UNDER"
                result["edge"] = under_edge_r
                result["hit_rate"] = under_hr
                result["best_line"] = best_under_o
            result["over_hit_rate"] = over_hr
            result["under_hit_rate"] = under_hr
            result["over_edge"] = over_edge_r
            result["under_edge"] = under_edge_r

            # Recalculate K distribution using final projected_ks
            from score_matchups import calc_k_distribution
            result["k_distribution"] = calc_k_distribution(result["projected_ks"])

            # Calculate hit rate, edge, and fair value for each ladder rung
            ladder_analysis = []
            seen_lines = set()
            for rung in odds.get("ladder", []):
                rung_line = rung["line"]
                seen_lines.add(rung_line)
                over_hr = calc_hit_rate(result["projected_ks"], rung_line)
                under_hr = round(100 - over_hr, 1)
                best_over = rung.get("best_over")
                best_under = rung.get("best_under")
                over_edge = calc_edge(over_hr, best_over["odds"]) if best_over else 0
                under_edge = calc_edge(under_hr, best_under["odds"]) if best_under else 0
                ladder_analysis.append({
                    "line": rung_line,
                    "over_hit_rate": over_hr,
                    "under_hit_rate": under_hr,
                    "over_edge": round(over_edge, 1),
                    "under_edge": round(under_edge, 1),
                    "best_over": best_over,
                    "best_under": best_under,
                    "fair_value_over": calc_fair_value_odds(over_hr),
                    "fair_value_under": calc_fair_value_odds(under_hr),
                    "model_only": False,
                })

            # Add model-only rungs for thresholds above the main line (no book needed)
            main_line = odds.get("line", 0)
            if main_line > 0:
                for offset in [1.0, 2.0, 3.0]:
                    ext_line = main_line + offset
                    if ext_line not in seen_lines:
                        over_hr = calc_hit_rate(result["projected_ks"], ext_line)
                        under_hr = round(100 - over_hr, 1)
                        ladder_analysis.append({
                            "line": ext_line,
                            "over_hit_rate": over_hr,
                            "under_hit_rate": under_hr,
                            "over_edge": 0,
                            "under_edge": 0,
                            "best_over": None,
                            "best_under": None,
                            "fair_value_over": calc_fair_value_odds(over_hr),
                            "fair_value_under": calc_fair_value_odds(under_hr),
                            "model_only": True,
                        })

            ladder_analysis.sort(key=lambda x: x["line"])
            result["ladder"] = ladder_analysis

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
            result["fair_value"] = calc_fair_value_odds(result.get("hit_rate", 50))

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
            edge_str = f"edge {result['edge']:+.1f}" if result['edge'] != 0 else "no odds"
            print(f" → {result['projected_ks']} Ks proj, {edge_str}")

    print(f"\n  Scored {len(scored)} pitcher matchups. ({time.time()-t0:.1f}s)")

    # Deduplicate by pitcher_id — doubleheaders can list the same starter for both games
    seen_pitchers: dict[int, int] = {}  # pitcher_id -> index in scored
    deduped = []
    for entry in scored:
        pid = entry.get("pitcher_id")
        if pid not in seen_pitchers:
            seen_pitchers[pid] = len(deduped)
            deduped.append(entry)
        else:
            # Keep whichever has the higher edge
            existing_idx = seen_pitchers[pid]
            if entry.get("edge", 0) > deduped[existing_idx].get("edge", 0):
                print(f"  [DEDUP] Replaced {entry.get('pitcher_name')} entry (higher edge kept)")
                deduped[existing_idx] = entry
            else:
                print(f"  [DEDUP] Dropped duplicate {entry.get('pitcher_name')} entry")
    scored = deduped

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
            "pitch_mix_adj": s.get("pitch_mix_adj", 1.0),
            "projected_ks_manual": s.get("projected_ks_manual", s.get("projected_ks", 0)),
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
            "pitcher_id": s.get("pitcher_id", 0),
            "pitcher_team": s.get("pitcher_team", ""),
            "pitcher_hand": s.get("pitcher_hand", "R"),
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
            "fair_value": s.get("fair_value", 0),
            "ladder": s.get("ladder", []),
            "last5": s.get("last5", []),
            "k_distribution": {
                str(k): v for k, v in s.get("k_distribution", {}).items()
            },
        }

    sorted_scored = sorted(scored, key=lambda x: x.get("edge", 0), reverse=True)
    strong = [make_play(s) for s in sorted_scored if s.get("edge", 0) >= EDGE_STRONG]
    moderate = [make_play(s) for s in sorted_scored if EDGE_MODERATE <= s.get("edge", 0) < EDGE_STRONG]
    lean = [make_play(s) for s in sorted_scored if EDGE_LEAN <= s.get("edge", 0) < EDGE_MODERATE]
    no_value = [make_play(s) for s in sorted_scored if s.get("edge", 0) < EDGE_LEAN]
    all_pitchers = [make_play(s) for s in sorted(scored, key=lambda x: x.get("projected_ks", 0), reverse=True)]

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
        "all_pitchers": all_pitchers,
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
