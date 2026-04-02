#!/usr/bin/env python3
"""
MLB Strikeout Sharp — Model Tuner & Calibration Report
Reads results.csv and analyzes projection accuracy, factor correlations,
bias patterns, and (once 50+ rows exist) trains an XGBoost model.

Usage: python3 tune_model.py
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pkl")

XGBOOST_THRESHOLD = 50  # minimum rows to train XGBoost

# Factor columns used for correlation analysis and XGBoost
FACTOR_COLS = [
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
]

# Current model weights (from score_matchups.py) for reference
CURRENT_WEIGHTS = {
    "swstr_pct → xk_rate": "SwStr% * 2.1 as base when no K% available",
    "csw_pct → xk_rate": "CSW * 0.75 blended at 35% weight",
    "k_pct": "65% weight in xk_rate blend",
    "lineup K% diff": "40% of lineup adjustment",
    "o_swing_pct diff": "25% of lineup adjustment",
    "swstr_pct_against diff": "20% of lineup adjustment",
    "z_contact_pct diff": "15% of lineup adjustment",
    "rolling form": "60/40 blend (model/form)",
}


def load_data() -> pd.DataFrame:
    """Load and clean results.csv."""
    if not os.path.exists(RESULTS_CSV):
        print("ERROR: results.csv not found. Run log_results.py first.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        print("ERROR: results.csv is empty. Log some game results first.")
        sys.exit(1)

    # Convert numeric columns
    for col in FACTOR_COLS + ["projected_ks", "actual_ks", "book_line", "our_edge_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def section_header(title: str, section_num: int):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  SECTION {section_num}: {title}")
    print(f"{'='*60}\n")


# =====================================================================
# SECTION 1: Calibration Report
# =====================================================================
def calibration_report(df: pd.DataFrame):
    section_header("CALIBRATION REPORT", 1)

    # Only analyze rows with projections
    has_proj = df.dropna(subset=["projected_ks", "actual_ks"])
    if has_proj.empty:
        print("  No rows with both projected and actual Ks found.")
        return

    print(f"  Total games logged: {len(df)}")
    print(f"  Games with projections: {len(has_proj)}")

    # Overall accuracy
    avg_proj = has_proj["projected_ks"].mean()
    avg_actual = has_proj["actual_ks"].mean()
    avg_miss = (has_proj["projected_ks"] - has_proj["actual_ks"]).mean()
    mae = (has_proj["projected_ks"] - has_proj["actual_ks"]).abs().mean()
    rmse = np.sqrt(((has_proj["projected_ks"] - has_proj["actual_ks"]) ** 2).mean())

    print(f"\n  Overall Accuracy:")
    print(f"    Avg Projected Ks:  {avg_proj:.2f}")
    print(f"    Avg Actual Ks:     {avg_actual:.2f}")
    print(f"    Avg Miss:          {avg_miss:+.2f} ({'over' if avg_miss > 0 else 'under'}-projecting)")
    print(f"    MAE:               {mae:.2f}")
    print(f"    RMSE:              {rmse:.2f}")

    # By edge tier
    has_edge = has_proj.dropna(subset=["our_edge_score"])
    if not has_edge.empty:
        tiers = [
            ("STRONG (edge > 8)", has_edge[has_edge["our_edge_score"] >= 8]),
            ("MODERATE (edge 4-8)", has_edge[(has_edge["our_edge_score"] >= 4) & (has_edge["our_edge_score"] < 8)]),
            ("LEAN (edge 1-4)", has_edge[(has_edge["our_edge_score"] >= 1) & (has_edge["our_edge_score"] < 4)]),
            ("NO VALUE (edge < 1)", has_edge[has_edge["our_edge_score"] < 1]),
        ]

        print(f"\n  {'Tier':<25} {'Plays':>6} {'Over%':>7} {'AvgProj':>8} {'AvgAct':>8} {'AvgMiss':>8}")
        print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")

        for tier_name, tier_df in tiers:
            if tier_df.empty:
                print(f"  {tier_name:<25} {'0':>6} {'—':>7} {'—':>8} {'—':>8} {'—':>8}")
                continue

            n = len(tier_df)
            has_line = tier_df.dropna(subset=["book_line"])
            has_line = has_line[has_line["book_line"] > 0]

            if not has_line.empty:
                over_pct = (has_line["actual_ks"] > has_line["book_line"]).mean() * 100
            else:
                over_pct = float("nan")

            avg_p = tier_df["projected_ks"].mean()
            avg_a = tier_df["actual_ks"].mean()
            miss = avg_p - avg_a

            over_str = f"{over_pct:.0f}%" if not np.isnan(over_pct) else "—"
            print(f"  {tier_name:<25} {n:>6} {over_str:>7} {avg_p:>8.1f} {avg_a:>8.1f} {miss:>+8.1f}")

    # Hit rate by over/under
    has_line = has_proj.dropna(subset=["book_line"])
    has_line = has_line[has_line["book_line"] > 0]
    if not has_line.empty:
        overs_hit = (has_line["actual_ks"] > has_line["book_line"]).sum()
        unders_hit = (has_line["actual_ks"] < has_line["book_line"]).sum()
        pushes = (has_line["actual_ks"] == has_line["book_line"]).sum()
        total = len(has_line)
        print(f"\n  Line Results:")
        print(f"    Overs hit:  {overs_hit}/{total} ({overs_hit/total*100:.0f}%)")
        print(f"    Unders hit: {unders_hit}/{total} ({unders_hit/total*100:.0f}%)")
        print(f"    Pushes:     {pushes}/{total} ({pushes/total*100:.0f}%)")


# =====================================================================
# SECTION 2: Factor Correlation
# =====================================================================
def factor_correlation(df: pd.DataFrame):
    section_header("FACTOR CORRELATION", 2)

    has_actual = df.dropna(subset=["actual_ks"])
    if has_actual.empty:
        print("  No actual K data to correlate against.")
        return

    print("  Pearson correlation of each factor with actual_ks:")
    print(f"  (+ = more of this factor → more Ks)\n")
    print(f"  {'Factor':<25} {'Corr':>8} {'p-value':>10} {'N':>6} {'Strength':<15}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*6} {'-'*15}")

    correlations = []
    for col in FACTOR_COLS:
        if col not in has_actual.columns:
            continue
        valid = has_actual.dropna(subset=[col, "actual_ks"])
        if len(valid) < 5:
            continue

        x = valid[col].astype(float)
        y = valid["actual_ks"].astype(float)

        if x.std() == 0:
            continue

        r, p = stats.pearsonr(x, y)
        correlations.append((col, r, p, len(valid)))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for col, r, p, n in correlations:
        if abs(r) >= 0.5:
            strength = "STRONG"
        elif abs(r) >= 0.3:
            strength = "Moderate"
        elif abs(r) >= 0.15:
            strength = "Weak"
        else:
            strength = "Negligible"

        sig = "*" if p < 0.05 else " "
        print(f"  {col:<25} {r:>+8.4f} {p:>10.4f}{sig} {n:>5} {strength:<15}")

    if correlations:
        top = correlations[0]
        print(f"\n  Strongest predictor: {top[0]} (r={top[1]:+.4f})")


# =====================================================================
# SECTION 3: Bias Detection
# =====================================================================
def bias_detection(df: pd.DataFrame):
    section_header("BIAS DETECTION", 3)

    has_both = df.dropna(subset=["projected_ks", "actual_ks"]).copy()
    if has_both.empty:
        print("  No data for bias analysis.")
        return

    has_both["miss"] = has_both["projected_ks"] - has_both["actual_ks"]

    print("  Checking for systematic over/under projection biases...\n")

    biases_found = []

    # Cold weather games (< 60°F)
    if "weather_temp" in has_both.columns:
        cold = has_both[has_both["weather_temp"].notna()]
        if not cold.empty:
            cold_games = cold[cold["weather_temp"] < 60]
            warm_games = cold[cold["weather_temp"] >= 60]
            if len(cold_games) >= 3 and len(warm_games) >= 3:
                cold_miss = cold_games["miss"].mean()
                warm_miss = warm_games["miss"].mean()
                diff = cold_miss - warm_miss
                biases_found.append({
                    "category": "Cold Weather (<60°F)",
                    "n": len(cold_games),
                    "avg_miss": cold_miss,
                    "vs_baseline": diff,
                    "direction": "over" if cold_miss > 0 else "under",
                })

    # High wind games (> 15 mph)
    if "weather_wind" in has_both.columns:
        wind = has_both[has_both["weather_wind"].notna()]
        if not wind.empty:
            high_wind = wind[wind["weather_wind"] > 15]
            low_wind = wind[wind["weather_wind"] <= 15]
            if len(high_wind) >= 3 and len(low_wind) >= 3:
                hw_miss = high_wind["miss"].mean()
                lw_miss = low_wind["miss"].mean()
                biases_found.append({
                    "category": "High Wind (>15 mph)",
                    "n": len(high_wind),
                    "avg_miss": hw_miss,
                    "vs_baseline": hw_miss - lw_miss,
                    "direction": "over" if hw_miss > 0 else "under",
                })

    # Short rest (<=4 days)
    if "days_rest" in has_both.columns:
        rest = has_both[has_both["days_rest"].notna()]
        if not rest.empty:
            short = rest[rest["days_rest"] <= 4]
            normal = rest[rest["days_rest"] > 4]
            if len(short) >= 3 and len(normal) >= 3:
                s_miss = short["miss"].mean()
                n_miss = normal["miss"].mean()
                biases_found.append({
                    "category": "Short Rest (<=4 days)",
                    "n": len(short),
                    "avg_miss": s_miss,
                    "vs_baseline": s_miss - n_miss,
                    "direction": "over" if s_miss > 0 else "under",
                })

    # By park (venues with 5+ games)
    if "venue" in has_both.columns:
        for venue, grp in has_both.groupby("venue"):
            if len(grp) >= 5:
                v_miss = grp["miss"].mean()
                if abs(v_miss) > 0.5:
                    biases_found.append({
                        "category": f"Park: {venue}",
                        "n": len(grp),
                        "avg_miss": v_miss,
                        "vs_baseline": v_miss - has_both["miss"].mean(),
                        "direction": "over" if v_miss > 0 else "under",
                    })

    # By umpire adjustment range
    if "ump_adjustment" in has_both.columns:
        ump = has_both[has_both["ump_adjustment"].notna()]
        if not ump.empty:
            expanded = ump[ump["ump_adjustment"] > 1.01]
            tight = ump[ump["ump_adjustment"] < 0.99]
            neutral = ump[(ump["ump_adjustment"] >= 0.99) & (ump["ump_adjustment"] <= 1.01)]

            for label, subset in [("Expanded Zone Umps", expanded),
                                  ("Tight Zone Umps", tight),
                                  ("Neutral Zone Umps", neutral)]:
                if len(subset) >= 3:
                    m = subset["miss"].mean()
                    biases_found.append({
                        "category": label,
                        "n": len(subset),
                        "avg_miss": m,
                        "vs_baseline": m - has_both["miss"].mean(),
                        "direction": "over" if m > 0 else "under",
                    })

    # High K% pitchers vs Low K% pitchers
    if "k_pct" in has_both.columns:
        kp = has_both[has_both["k_pct"].notna() & (has_both["k_pct"] > 0)]
        if len(kp) >= 10:
            median_k = kp["k_pct"].median()
            high_k = kp[kp["k_pct"] >= median_k]
            low_k = kp[kp["k_pct"] < median_k]
            if len(high_k) >= 3 and len(low_k) >= 3:
                hk_miss = high_k["miss"].mean()
                lk_miss = low_k["miss"].mean()
                biases_found.append({
                    "category": "High K% Pitchers (above median)",
                    "n": len(high_k),
                    "avg_miss": hk_miss,
                    "vs_baseline": hk_miss - lk_miss,
                    "direction": "over" if hk_miss > 0 else "under",
                })
                biases_found.append({
                    "category": "Low K% Pitchers (below median)",
                    "n": len(low_k),
                    "avg_miss": lk_miss,
                    "vs_baseline": lk_miss - hk_miss,
                    "direction": "over" if lk_miss > 0 else "under",
                })

    if not biases_found:
        print("  Not enough data to detect biases yet (need 3+ games per category).")
        return

    # Sort by magnitude of bias
    biases_found.sort(key=lambda x: abs(x["avg_miss"]), reverse=True)

    print(f"  {'Category':<35} {'N':>4} {'AvgMiss':>9} {'vs Base':>9} {'Direction':<10}")
    print(f"  {'-'*35} {'-'*4} {'-'*9} {'-'*9} {'-'*10}")

    for b in biases_found:
        flag = " ⚠" if abs(b["avg_miss"]) > 1.0 else ""
        print(f"  {b['category']:<35} {b['n']:>4} {b['avg_miss']:>+9.2f} "
              f"{b['vs_baseline']:>+9.2f} {b['direction']:<10}{flag}")

    # Flag significant biases
    sig_biases = [b for b in biases_found if abs(b["avg_miss"]) > 1.0]
    if sig_biases:
        print(f"\n  ⚠ Found {len(sig_biases)} significant biases (>1.0 K avg miss):")
        for b in sig_biases:
            direction = "over" if b["avg_miss"] > 0 else "under"
            print(f"    → {b['category']}: {direction}-projecting by {abs(b['avg_miss']):.1f} Ks on average")


# =====================================================================
# SECTION 4: Weight Suggestions
# =====================================================================
def weight_suggestions(df: pd.DataFrame):
    section_header("WEIGHT SUGGESTIONS", 4)

    has_both = df.dropna(subset=["projected_ks", "actual_ks"]).copy()
    if len(has_both) < 10:
        print(f"  Need at least 10 logged results for weight suggestions.")
        print(f"  Current: {len(has_both)} rows. Log {10 - len(has_both)} more games.")
        return

    has_both["miss"] = has_both["projected_ks"] - has_both["actual_ks"]
    overall_miss = has_both["miss"].mean()

    print("  Based on correlation data and projection accuracy:\n")

    # Calculate correlations
    correlations = {}
    for col in FACTOR_COLS:
        if col not in has_both.columns:
            continue
        valid = has_both.dropna(subset=[col])
        if len(valid) < 5:
            continue
        x = valid[col].astype(float)
        y = valid["actual_ks"].astype(float)
        if x.std() == 0:
            continue
        r, _ = stats.pearsonr(x, y)
        correlations[col] = r

    if not correlations:
        print("  Not enough factor data for weight suggestions.")
        return

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    suggestions = []

    # Overall bias suggestion
    if abs(overall_miss) > 0.5:
        if overall_miss > 0:
            suggestions.append(
                f"OVERALL BIAS: You are over-projecting by {overall_miss:.1f} Ks on average. "
                f"Consider reducing DEFAULT_BATTERS_FACED in config.py from 25 to "
                f"{max(20, 25 - int(overall_miss * 2))} or reducing the form blend weight."
            )
        else:
            suggestions.append(
                f"OVERALL BIAS: You are under-projecting by {abs(overall_miss):.1f} Ks on average. "
                f"Consider increasing DEFAULT_BATTERS_FACED in config.py from 25 to "
                f"{min(30, 25 + int(abs(overall_miss) * 2))}."
            )

    # Top factor suggestions
    for col, r in sorted_corr[:5]:
        if abs(r) >= 0.4:
            col_display = col.replace("_", " ").replace("pct", "%").title()
            suggestions.append(
                f"{col_display} is your strongest predictor (r={r:+.3f}). "
                f"Consider increasing its weight in the scoring model."
            )
        elif abs(r) >= 0.2:
            col_display = col.replace("_", " ").replace("pct", "%").title()
            suggestions.append(
                f"{col_display} shows moderate correlation (r={r:+.3f}). "
                f"Current weight seems reasonable — monitor as more data comes in."
            )
        elif abs(r) < 0.1:
            col_display = col.replace("_", " ").replace("pct", "%").title()
            suggestions.append(
                f"{col_display} shows negligible correlation (r={r:+.3f}). "
                f"Consider reducing its weight or removing it from the model."
            )

    # Specific weight adjustment suggestions based on correlation rankings
    if len(sorted_corr) >= 2:
        top = sorted_corr[0]
        second = sorted_corr[1]
        if abs(top[1]) > abs(second[1]) * 1.5:
            suggestions.append(
                f"\nKEY INSIGHT: {top[0]} is significantly more predictive than {second[0]}. "
                f"The model should lean more heavily on {top[0]}."
            )

    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}\n")

    if not suggestions:
        print("  Not enough differentiation in correlations yet. Keep logging games.")


# =====================================================================
# SECTION 5: XGBoost Auto-Tuning
# =====================================================================
def xgboost_tuning(df: pd.DataFrame):
    section_header("XGBOOST AUTO-TUNING", 5)

    has_both = df.dropna(subset=["projected_ks", "actual_ks"])

    # Check available factor columns
    available_factors = [c for c in FACTOR_COLS if c in has_both.columns]
    has_factors = has_both.dropna(subset=available_factors)

    if len(has_factors) < XGBOOST_THRESHOLD:
        remaining = XGBOOST_THRESHOLD - len(has_factors)
        filled = len(has_factors)
        bar_len = 30
        filled_bars = int((filled / XGBOOST_THRESHOLD) * bar_len)
        bar = "█" * filled_bars + "░" * (bar_len - filled_bars)

        print(f"  XGBoost auto-tuning requires {XGBOOST_THRESHOLD} complete rows.")
        print(f"  Current: {filled}/{XGBOOST_THRESHOLD}")
        print(f"  [{bar}] {filled/XGBOOST_THRESHOLD*100:.0f}%")
        print(f"\n  Log {remaining} more game results to unlock auto-tuning.")
        print(f"  At ~12 starters/day, that's roughly {remaining // 12 + 1} more days of games.")
        return

    try:
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    except ImportError:
        print("  XGBoost or scikit-learn not installed.")
        print("  Install with: pip3 install xgboost scikit-learn")
        return

    print(f"  Training XGBoost on {len(has_factors)} rows with {len(available_factors)} features...\n")

    X = has_factors[available_factors].astype(float)
    y = has_factors["actual_ks"].astype(float)

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 5), scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    # Train on full data
    model.fit(X, y)

    # Predictions on training data (for comparison)
    y_pred = model.predict(X)
    train_mae = mean_absolute_error(y, y_pred)
    train_rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Compare to manual model
    manual_mae = (has_factors["projected_ks"] - has_factors["actual_ks"]).abs().mean()
    manual_rmse = np.sqrt(((has_factors["projected_ks"] - has_factors["actual_ks"]) ** 2).mean())

    print(f"  Model Performance:")
    print(f"    {'Metric':<20} {'Manual Model':>15} {'XGBoost':>15} {'Improvement':>15}")
    print(f"    {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
    print(f"    {'MAE':<20} {manual_mae:>15.2f} {train_mae:>15.2f} {(manual_mae-train_mae)/manual_mae*100:>+14.1f}%")
    print(f"    {'RMSE':<20} {manual_rmse:>15.2f} {train_rmse:>15.2f} {(manual_rmse-train_rmse)/manual_rmse*100:>+14.1f}%")
    print(f"    {'CV MAE (5-fold)':<20} {'—':>15} {cv_mae:>15.2f} {'—':>15}")

    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(available_factors, importance), key=lambda x: x[1], reverse=True)

    print(f"\n  Feature Importance Rankings:")
    print(f"    {'Rank':>4}  {'Feature':<25} {'Importance':>12} {'Bar'}")
    print(f"    {'-'*4}  {'-'*25} {'-'*12} {'-'*20}")

    max_imp = feat_imp[0][1] if feat_imp else 1
    for rank, (feat, imp) in enumerate(feat_imp, 1):
        bar_len = int((imp / max_imp) * 20)
        bar = "█" * bar_len
        print(f"    {rank:>4}  {feat:<25} {imp:>12.4f} {bar}")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "features": available_factors,
            "train_rows": len(has_factors),
            "train_mae": train_mae,
            "cv_mae": cv_mae,
        }, f)
    print(f"\n  Model saved to {MODEL_PATH}")
    print(f"  To use in run.py, load with: pickle.load(open('model.pkl', 'rb'))")

    # Suggest weight adjustments based on feature importance
    print(f"\n  Implied Weight Distribution (from XGBoost):")
    total_imp = sum(imp for _, imp in feat_imp)
    for feat, imp in feat_imp[:7]:
        pct = (imp / total_imp) * 100
        feat_display = feat.replace("_", " ").replace("pct", "%").title()
        print(f"    {feat_display}: {pct:.1f}% of prediction weight")


# =====================================================================
# Main
# =====================================================================
def main():
    print(f"\n{'='*60}")
    print(f"  MLB STRIKEOUT SHARP — MODEL TUNER")
    print(f"{'='*60}")

    df = load_data()
    print(f"\n  Loaded {len(df)} rows from results.csv")

    # Date range
    if "date" in df.columns:
        dates = df["date"].dropna().unique()
        print(f"  Date range: {min(dates)} to {max(dates)} ({len(dates)} game days)")

    calibration_report(df)
    factor_correlation(df)
    bias_detection(df)
    weight_suggestions(df)
    xgboost_tuning(df)

    print(f"\n{'='*60}")
    print(f"  TUNING COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
