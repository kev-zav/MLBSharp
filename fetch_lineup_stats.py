"""
Fetch opposing team/lineup strikeout vulnerability stats from Baseball Savant.
Uses Statcast pitch-level data — pulled directly from MLB, never blocked.
Calculates K%, Whiff%, Chase rate, Z-Contact% broken down by pitcher handedness.
"""

import warnings
from datetime import date

import pandas as pd
from pybaseball import statcast
from config import SEASON, LEAGUE_AVG_K_PCT, LEAGUE_AVG_SWSTR

warnings.filterwarnings("ignore")

_cache: dict = {}

_DEFAULTS = {
    "team_k_pct": LEAGUE_AVG_K_PCT,
    "team_k_pct_vs_rhp": LEAGUE_AVG_K_PCT,
    "team_k_pct_vs_lhp": LEAGUE_AVG_K_PCT,
    "o_swing_pct": 0.30,
    "swstr_pct_against": LEAGUE_AVG_SWSTR,
    "whiff_pct": LEAGUE_AVG_SWSTR,
    "z_contact_pct": 0.82,
    "team_k_pct_last10": LEAGUE_AVG_K_PCT,
    "avg_batter_k_pct": LEAGUE_AVG_K_PCT,
    "batter_k_pcts": [],
    "bb_pct": 0.085,
    "chase_by_pitch_type": {},
}


def _get_season_statcast() -> pd.DataFrame:
    """Pull all Statcast pitch data since season start. Cached per session."""
    cache_key = f"statcast_season_{SEASON}"
    if cache_key in _cache:
        return _cache[cache_key]

    season_start = date(SEASON, 3, 18)  # covers opening day regardless of year
    end = date.today()

    print(f"  [LINEUP] Pulling Statcast data {season_start} → {end}...")
    try:
        df = statcast(
            start_dt=season_start.strftime("%Y-%m-%d"),
            end_dt=end.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        print(f"  [LINEUP] Statcast pull failed: {e}")
        df = pd.DataFrame()

    if not df.empty:
        print(f"  [LINEUP] {len(df):,} pitches loaded.")

    _cache[cache_key] = df
    return df


def _build_team_stats(df: pd.DataFrame) -> dict:
    """
    Aggregate pitch-level data into per-team, per-pitcher-handedness stats.
    Returns dict keyed by (team_abbr, p_throws).
    """
    if df.empty:
        return {}

    df = df.copy()

    # Derive batting team abbreviation from inning half + home/away
    df["batting_team"] = df.apply(
        lambda r: r["away_team"] if r.get("inning_topbot") == "Top" else r["home_team"],
        axis=1,
    )

    swing_desc = {
        "swinging_strike", "swinging_strike_blocked", "foul_tip",
        "foul", "foul_bunt", "hit_into_play",
    }
    miss_desc = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
    contact_desc = {"foul", "foul_bunt", "hit_into_play"}
    in_zone = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    out_zone = {11, 12, 13, 14}

    results = {}
    pitch_type_whiff = {}  # keyed by (team_abbr, pitch_type)
    pitch_type_chase = {}  # keyed by (team_abbr, pitch_type)

    # Build team whiff rate and chase rate by pitch type
    for (bat_team, pitch_type), pt_grp in df.groupby(["batting_team", "pitch_type"]):
        if not pitch_type or str(pitch_type) == "nan":
            continue
        pt_swings = pt_grp[pt_grp["description"].isin(swing_desc)]
        pt_misses = pt_grp[pt_grp["description"].isin(miss_desc)]
        if len(pt_swings) < 15:  # need at least 15 swings for reliability
            continue
        raw_whiff = len(pt_misses) / len(pt_swings)
        # Regress toward league average (swing count based)
        reg_w = len(pt_swings) / (len(pt_swings) + 50)
        pitch_type_whiff[(str(bat_team), str(pitch_type))] = round(
            reg_w * raw_whiff + (1 - reg_w) * LEAGUE_AVG_SWSTR, 4
        )

        # Chase rate by pitch type (swings on out-of-zone pitches / total out-of-zone pitches)
        pt_out_zone = pt_grp[pt_grp["zone"].isin(out_zone)]
        pt_out_zone_swings = pt_out_zone["description"].isin(swing_desc).sum()
        pt_out_zone_pitches = len(pt_out_zone)
        if pt_out_zone_pitches > 0:
            raw_chase = pt_out_zone_swings / pt_out_zone_pitches
            swings = pt_out_zone_swings
            chase_reg_w = swings / (swings + 50)
            pitch_type_chase[(str(bat_team), str(pitch_type))] = round(
                chase_reg_w * raw_chase + (1 - chase_reg_w) * 0.30, 4
            )

    for (bat_team, p_hand), grp in df.groupby(["batting_team", "p_throws"]):
        # Plate appearances = rows where events is populated
        pa = grp[grp["events"].notna() & (grp["events"] != "")]
        pa_count = len(pa)
        if pa_count < 10:
            continue

        # K%
        k_pct = len(pa[pa["events"] == "strikeout"]) / pa_count

        # Whiff% (misses / total swings)
        is_swing = grp["description"].isin(swing_desc)
        is_miss = grp["description"].isin(miss_desc)
        total_swings = is_swing.sum()
        whiff_pct = is_miss.sum() / total_swings if total_swings > 0 else LEAGUE_AVG_SWSTR

        # Chase rate (swings on pitches outside zone / total pitches outside zone)
        outside = grp[grp["zone"].isin(out_zone)]
        outside_swings = outside["description"].isin(swing_desc).sum()
        chase_pct = outside_swings / len(outside) if len(outside) > 0 else 0.30

        # Z-Contact% (contact on pitches in zone / swings on pitches in zone)
        in_z = grp[grp["zone"].isin(in_zone)]
        z_swings = in_z["description"].isin(swing_desc).sum()
        z_contact = in_z["description"].isin(contact_desc).sum()
        z_contact_pct = z_contact / z_swings if z_swings > 0 else 0.82

        # Regress each rate toward league average weighted by PA
        # At 50 PA: ~33% real data. At 150 PA: ~75%. At 300 PA: ~86%.
        regression_pa = 100
        w = pa_count / (pa_count + regression_pa)
        reg_k_pct = w * k_pct + (1 - w) * LEAGUE_AVG_K_PCT
        reg_whiff = w * whiff_pct + (1 - w) * LEAGUE_AVG_SWSTR
        reg_chase = w * chase_pct + (1 - w) * 0.30
        reg_z_contact = w * z_contact_pct + (1 - w) * 0.82

        # BB% (walk rate) — regress toward league average of 8.5%
        bb_count = len(pa[pa["events"] == "walk"])
        bb_pct = bb_count / pa_count
        reg_bb_pct = w * bb_pct + (1 - w) * 0.085

        # Collect pitch type whiff rates for this team
        team_pt_whiff = {
            pt: whiff for (team, pt), whiff in pitch_type_whiff.items()
            if team == str(bat_team)
        }

        # Collect pitch type chase rates for this team
        team_pt_chase = {
            pt: chase for (team, pt), chase in pitch_type_chase.items()
            if team == str(bat_team)
        }

        results[(str(bat_team), str(p_hand))] = {
            "team_k_pct": round(reg_k_pct, 4),
            "o_swing_pct": round(reg_chase, 4),
            "whiff_pct": round(reg_whiff, 4),
            "swstr_pct_against": round(reg_whiff, 4),
            "z_contact_pct": round(reg_z_contact, 4),
            "pa_count": pa_count,
            "whiff_by_pitch_type": team_pt_whiff,
            "bb_pct": round(reg_bb_pct, 4),
            "chase_by_pitch_type": team_pt_chase,
        }

    return results


def _get_team_stats_cache() -> dict:
    """Return (or build) the season team stats lookup."""
    cache_key = f"team_stats_{SEASON}"
    if cache_key not in _cache:
        df = _get_season_statcast()
        _cache[cache_key] = _build_team_stats(df)
    return _cache[cache_key]


def fetch_lineup_stats(team_abbr: str, pitcher_hand: str = "R") -> dict:
    """
    Main entry point. Returns team K vulnerability profile vs a given pitcher hand.
    Falls back to league averages if data is unavailable.
    """
    team_stats = _get_team_stats_cache()

    result = team_stats.get((team_abbr, pitcher_hand))

    # Fall back to opposite hand if this split has no data yet
    if result is None:
        alt_hand = "L" if pitcher_hand == "R" else "R"
        result = team_stats.get((team_abbr, alt_hand))

    if result is None:
        return {**_DEFAULTS, "pitcher_hand": pitcher_hand}

    k_pct = result["team_k_pct"]
    return {
        **_DEFAULTS,
        **result,
        "team_k_pct_vs_rhp": k_pct if pitcher_hand == "R" else _DEFAULTS["team_k_pct_vs_rhp"],
        "team_k_pct_vs_lhp": k_pct if pitcher_hand == "L" else _DEFAULTS["team_k_pct_vs_lhp"],
        "team_k_pct_last10": k_pct,
        "avg_batter_k_pct": k_pct,
        "batter_k_pcts": [],
        "pitcher_hand": pitcher_hand,
        "whiff_by_pitch_type": result.get("whiff_by_pitch_type", {}),
        "bb_pct": result.get("bb_pct", 0.085),
        "chase_by_pitch_type": result.get("chase_by_pitch_type", {}),
    }


if __name__ == "__main__":
    for team in ["NYY", "BOS", "LAD", "HOU"]:
        stats = fetch_lineup_stats(team, "R")
        print(f"\n{team} vs RHP (PA: {stats.get('pa_count', '?')}):")
        for k, v in stats.items():
            if k in ("batter_k_pcts", "pitcher_hand", "pa_count"):
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.1%}")
