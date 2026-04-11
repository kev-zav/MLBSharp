"""
Fetch per-batter Statcast stats for a given lineup.
Computes K%, BB%, whiff%, chase%, z-contact%, and pitch-type vulnerability
for each batter, then aggregates across the lineup.

Reuses the season-level Statcast DataFrame already pulled by fetch_lineup_stats
to avoid redundant API calls.
"""

from datetime import date, timedelta

import pandas as pd
from config import LEAGUE_AVG_K_PCT, LEAGUE_AVG_BB_PCT, LEAGUE_AVG_SWSTR

_SWING_DESC = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
    "foul", "foul_bunt", "hit_into_play",
}
_MISS_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
_CONTACT_DESC = {"foul", "foul_bunt", "hit_into_play"}
_IN_ZONE = {1, 2, 3, 4, 5, 6, 7, 8, 9}
_OUT_ZONE = {11, 12, 13, 14}

# Regression denominator: equivalent PA worth of league average to blend in
_REGRESSION_PA = 100


def _regress(raw: float, pa: int, league_avg: float) -> float:
    w = pa / (pa + _REGRESSION_PA)
    return round(w * raw + (1 - w) * league_avg, 4)


def _compute_rates(df: pd.DataFrame, pa_count: int) -> dict:
    """
    Compute raw (unregressed) whiff%, chase%, z-contact%, and per-pitch-type rates
    from a Statcast DataFrame slice. Returns dict of raw values + swing count.
    """
    is_swing = df["description"].isin(_SWING_DESC)
    is_miss = df["description"].isin(_MISS_DESC)
    total_swings = is_swing.sum()
    whiff_pct = is_miss.sum() / total_swings if total_swings > 0 else LEAGUE_AVG_SWSTR

    outside = df[df["zone"].isin(_OUT_ZONE)] if "zone" in df.columns else pd.DataFrame()
    if len(outside) > 0:
        outside_swings = outside["description"].isin(_SWING_DESC).sum()
        chase_pct = outside_swings / len(outside)
    else:
        chase_pct = 0.30

    in_z = df[df["zone"].isin(_IN_ZONE)] if "zone" in df.columns else pd.DataFrame()
    if len(in_z) > 0:
        z_swings = in_z["description"].isin(_SWING_DESC).sum()
        z_contact = in_z["description"].isin(_CONTACT_DESC).sum()
        z_contact_pct = z_contact / z_swings if z_swings > 0 else 0.82
    else:
        z_contact_pct = 0.82

    whiff_by_pitch: dict[str, float] = {}
    chase_by_pitch: dict[str, float] = {}
    if "pitch_type" in df.columns:
        swings_df = df[df["description"].isin(_SWING_DESC)]
        for pt, grp in swings_df.groupby("pitch_type"):
            if not pt or str(pt) == "nan" or len(grp) < 10:
                continue
            misses = grp[grp["description"].isin(_MISS_DESC)]
            whiff_by_pitch[str(pt)] = len(misses) / len(grp)

        if "zone" in df.columns:
            for pt, grp in df.groupby("pitch_type"):
                if not pt or str(pt) == "nan":
                    continue
                out_z = grp[grp["zone"].isin(_OUT_ZONE)]
                if len(out_z) < 10:
                    continue
                swings_oz = out_z["description"].isin(_SWING_DESC).sum()
                chase_by_pitch[str(pt)] = swings_oz / len(out_z)

    return {
        "whiff_pct": whiff_pct,
        "total_swings": int(total_swings),
        "chase_pct": chase_pct,
        "z_contact_pct": z_contact_pct,
        "whiff_by_pitch_type": whiff_by_pitch,
        "chase_by_pitch_type": chase_by_pitch,
    }


def _blend(season_rate: float, recent_rate: float, recent_pa: int) -> float:
    """
    Blend season rate with recent rate. Weight scales from ~0 at 0 PA to max 30% at 50+ PA.
    Formula: min(0.30, recent_pa / (recent_pa + 120))
    """
    trend_weight = min(0.30, recent_pa / (recent_pa + 120))
    return season_rate * (1 - trend_weight) + recent_rate * trend_weight


def calc_batter_stats(batter_df: pd.DataFrame, pitcher_hand: str) -> dict | None:
    """
    Compute stats for a single batter from their Statcast rows filtered to a given pitcher hand.
    Season rate is the base; last 21 days provide a directional trend nudge (max 30% weight).
    Returns None if insufficient data (<10 PA).
    """
    # Filter to the relevant pitcher handedness
    df = batter_df[batter_df["p_throws"] == pitcher_hand] if "p_throws" in batter_df.columns else batter_df

    pa = df[df["events"].notna() & (df["events"] != "")]
    pa_count = len(pa)
    if pa_count < 10:
        return None

    # K% and BB%
    k_pct = len(pa[pa["events"] == "strikeout"]) / pa_count
    bb_pct = len(pa[pa["events"] == "walk"]) / pa_count

    # Season-level rates
    season = _compute_rates(df, pa_count)

    # Recent trend (last 21 days) — same pitcher-hand filter already applied
    recent_df = pd.DataFrame()
    if "game_date" in df.columns:
        cutoff = (date.today() - timedelta(days=21)).strftime("%Y-%m-%d")
        try:
            recent_df = df[df["game_date"] >= cutoff]
        except Exception:
            recent_df = pd.DataFrame()

    recent_pa_rows = recent_df[recent_df["events"].notna() & (recent_df["events"] != "")] if not recent_df.empty else pd.DataFrame()
    recent_pa = len(recent_pa_rows)

    if recent_pa >= 5:
        recent = _compute_rates(recent_df, recent_pa)

        whiff_pct = _blend(season["whiff_pct"], recent["whiff_pct"], recent_pa)
        chase_pct = _blend(season["chase_pct"], recent["chase_pct"], recent_pa)
        z_contact_pct = _blend(season["z_contact_pct"], recent["z_contact_pct"], recent_pa)

        # Per-pitch-type blend: only blend where recent has enough swings; fall back to season
        whiff_by_pitch = dict(season["whiff_by_pitch_type"])
        for pt, recent_rate in recent["whiff_by_pitch_type"].items():
            season_rate = season["whiff_by_pitch_type"].get(pt, recent_rate)
            whiff_by_pitch[pt] = round(_blend(season_rate, recent_rate, recent_pa), 4)

        chase_by_pitch = dict(season["chase_by_pitch_type"])
        for pt, recent_rate in recent["chase_by_pitch_type"].items():
            season_rate = season["chase_by_pitch_type"].get(pt, recent_rate)
            chase_by_pitch[pt] = round(_blend(season_rate, recent_rate, recent_pa), 4)
    else:
        whiff_pct = season["whiff_pct"]
        chase_pct = season["chase_pct"]
        z_contact_pct = season["z_contact_pct"]
        whiff_by_pitch = {pt: round(v, 4) for pt, v in season["whiff_by_pitch_type"].items()}
        chase_by_pitch = {pt: round(v, 4) for pt, v in season["chase_by_pitch_type"].items()}

    return {
        "pa_count": pa_count,
        "recent_pa": recent_pa,
        "k_pct": _regress(k_pct, pa_count, LEAGUE_AVG_K_PCT),
        "bb_pct": _regress(bb_pct, pa_count, LEAGUE_AVG_BB_PCT),
        "whiff_pct": _regress(whiff_pct, season["total_swings"], LEAGUE_AVG_SWSTR),
        "chase_pct": _regress(chase_pct, pa_count, 0.30),
        "z_contact_pct": _regress(z_contact_pct, pa_count, 0.82),
        "whiff_by_pitch_type": whiff_by_pitch,
        "chase_by_pitch_type": chase_by_pitch,
    }


def aggregate_lineup(batter_stats_list: list[dict]) -> dict:
    """
    Weighted average of per-batter stats across the lineup.
    Weights by PA count so batters with more data carry more influence.
    Returns a dict shaped like fetch_lineup_stats output.
    """
    if not batter_stats_list:
        return {}

    total_pa = sum(b["pa_count"] for b in batter_stats_list)
    if total_pa == 0:
        return {}

    def wavg(key: str, default: float) -> float:
        return round(
            sum(b.get(key, default) * b["pa_count"] for b in batter_stats_list) / total_pa,
            4,
        )

    # Aggregate whiff_by_pitch_type: weighted average per pitch type
    all_pitch_types = set()
    for b in batter_stats_list:
        all_pitch_types.update(b.get("whiff_by_pitch_type", {}).keys())

    whiff_by_pitch = {}
    chase_by_pitch = {}
    for pt in all_pitch_types:
        pt_pa = [(b["pa_count"], b["whiff_by_pitch_type"].get(pt)) for b in batter_stats_list if pt in b.get("whiff_by_pitch_type", {})]
        if pt_pa:
            total = sum(pa for pa, _ in pt_pa)
            whiff_by_pitch[pt] = round(sum(pa * v for pa, v in pt_pa) / total, 4)

        pt_chase = [(b["pa_count"], b["chase_by_pitch_type"].get(pt)) for b in batter_stats_list if pt in b.get("chase_by_pitch_type", {})]
        if pt_chase:
            total = sum(pa for pa, _ in pt_chase)
            chase_by_pitch[pt] = round(sum(pa * v for pa, v in pt_chase) / total, 4)

    total_recent_pa = sum(b.get("recent_pa", 0) for b in batter_stats_list)

    return {
        "team_k_pct": wavg("k_pct", LEAGUE_AVG_K_PCT),
        "o_swing_pct": wavg("chase_pct", 0.30),
        "whiff_pct": wavg("whiff_pct", LEAGUE_AVG_SWSTR),
        "swstr_pct_against": wavg("whiff_pct", LEAGUE_AVG_SWSTR),
        "z_contact_pct": wavg("z_contact_pct", 0.82),
        "bb_pct": wavg("bb_pct", LEAGUE_AVG_BB_PCT),
        "pa_count": total_pa,
        "recent_pa": total_recent_pa,
        "whiff_by_pitch_type": whiff_by_pitch,
        "chase_by_pitch_type": chase_by_pitch,
        "batter_count": len(batter_stats_list),
    }


def fetch_batter_lineup_stats(
    batter_ids: list[int],
    pitcher_hand: str,
    season_df: pd.DataFrame,
) -> dict | None:
    """
    Main entry point. Given a list of batter IDs, pitcher hand, and the full
    season Statcast DataFrame (already loaded), compute and aggregate lineup stats.

    Returns aggregated dict or None if insufficient data (fewer than 3 batters with data).
    """
    if season_df.empty or not batter_ids:
        return None

    batter_stats = []
    for bid in batter_ids:
        batter_df = season_df[season_df["batter"] == bid]
        if batter_df.empty:
            continue
        stats = calc_batter_stats(batter_df, pitcher_hand)
        if stats:
            batter_stats.append(stats)

    if len(batter_stats) < 3:
        return None  # not enough individual data, fall back to team aggregate

    return aggregate_lineup(batter_stats)
