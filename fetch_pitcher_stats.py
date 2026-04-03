"""
Fetch pitcher Statcast and FanGraphs stats via pybaseball.
Returns SwStr%, CSW%, K%, K/9, Whiff% by pitch, velocity trends,
rolling K averages, pitch count, and days rest.
"""

import warnings
from datetime import date, timedelta

import pandas as pd
from pybaseball import (
    playerid_lookup,
    statcast_pitcher,
    pitching_stats,
)
from config import SEASON, LEAGUE_AVG_SWSTR

warnings.filterwarnings("ignore")

# Cache to avoid redundant pybaseball calls within a session
_cache: dict[str, any] = {}


def _mlb_id_to_fangraphs_name(pitcher_name: str) -> tuple[str, str]:
    """Split 'First Last' into (last, first) for playerid_lookup."""
    parts = pitcher_name.strip().split()
    if len(parts) >= 2:
        return parts[-1], parts[0]
    return parts[0], ""


def get_statcast_data(pitcher_id: int, days_back: int = 60) -> pd.DataFrame:
    """Pull recent Statcast pitch-level data for a pitcher."""
    cache_key = f"sc_{pitcher_id}_{days_back}"
    if cache_key in _cache:
        return _cache[cache_key]

    end = date.today()
    start = end - timedelta(days=days_back)
    try:
        df = statcast_pitcher(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            pitcher_id,
        )
    except Exception:
        df = pd.DataFrame()

    _cache[cache_key] = df
    return df


def calc_swstr_pct(df: pd.DataFrame) -> float:
    """Swinging strike rate from Statcast data."""
    if df.empty:
        return LEAGUE_AVG_SWSTR
    total = len(df)
    swinging_strikes = len(df[df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
        "foul_tip",  # foul tips are Ks
    ])])
    return round(swinging_strikes / total, 4) if total else LEAGUE_AVG_SWSTR


def calc_csw_pct(df: pd.DataFrame) -> float:
    """Called strikes + whiffs per pitch."""
    if df.empty:
        return 0.0
    total = len(df)
    csw = len(df[df["description"].isin([
        "called_strike", "swinging_strike", "swinging_strike_blocked",
        "foul_tip",
    ])])
    return round(csw / total, 4) if total else 0.0


def calc_whiff_by_pitch(df: pd.DataFrame) -> dict[str, float]:
    """Whiff% broken down by pitch type."""
    if df.empty:
        return {}
    results = {}
    swings = df[df["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul_tip",
        "foul", "foul_bunt", "hit_into_play",
    ])]
    for pitch_type, grp in swings.groupby("pitch_type"):
        whiffs = grp[grp["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul_tip",
        ])]
        pct = round(len(whiffs) / len(grp), 4) if len(grp) > 0 else 0.0
        results[pitch_type] = pct
    return results


def calc_velocity_trend(df: pd.DataFrame) -> dict:
    """Average fastball velocity over last 5 game dates."""
    if df.empty:
        return {"avg_velo": 0.0, "trend": 0.0}

    fb_types = ["FF", "SI", "FC"]
    fb = df[df["pitch_type"].isin(fb_types)].copy()
    if fb.empty:
        return {"avg_velo": 0.0, "trend": 0.0}

    fb["game_date"] = pd.to_datetime(fb["game_date"])
    dates = sorted(fb["game_date"].unique())[-5:]
    fb_recent = fb[fb["game_date"].isin(dates)]

    avg_velo = round(fb_recent["release_speed"].mean(), 1)

    # Trend: compare last 2 games vs first 2 games in window
    if len(dates) >= 4:
        early = fb[fb["game_date"].isin(dates[:2])]["release_speed"].mean()
        late = fb[fb["game_date"].isin(dates[-2:])]["release_speed"].mean()
        trend = round(late - early, 1)
    else:
        trend = 0.0

    return {"avg_velo": avg_velo, "trend": trend}


def get_game_logs(df: pd.DataFrame) -> list[dict]:
    """Extract per-game K totals and pitch counts from Statcast data."""
    if df.empty:
        return []

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    logs = []

    for gdate, grp in df.groupby("game_date"):
        ks = len(grp[grp["events"] == "strikeout"])
        pitches = len(grp)
        logs.append({
            "date": gdate,
            "strikeouts": ks,
            "pitches": pitches,
        })

    logs.sort(key=lambda x: x["date"])
    return logs


def calc_rolling_ks(game_logs: list[dict], n: int) -> float:
    """Average Ks over last n starts."""
    if not game_logs or len(game_logs) == 0:
        return 0.0
    recent = game_logs[-n:]
    return round(sum(g["strikeouts"] for g in recent) / len(recent), 2)


def calc_swstr_trend(df: pd.DataFrame) -> float:
    """SwStr% over last 3 starts minus season SwStr%. Positive = improving."""
    if df.empty:
        return 0.0
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    dates = sorted(df["game_date"].unique())

    season_swstr = calc_swstr_pct(df)

    last3_dates = dates[-3:] if len(dates) >= 3 else dates
    last3 = df[df["game_date"].isin(last3_dates)]
    last3_swstr = calc_swstr_pct(last3)

    return round(last3_swstr - season_swstr, 4)


def get_fangraphs_stats(pitcher_name: str) -> dict:
    """Pull season-level K%, K/9 from FanGraphs via pybaseball."""
    cache_key = f"fg_{pitcher_name}_{SEASON}"
    if cache_key in _cache:
        return _cache[cache_key]

    defaults = {"k_pct": 0.0, "k_per_9": 0.0, "xk_pct": 0.0}
    try:
        stats = pitching_stats(SEASON, SEASON, qual=1)
        if stats.empty:
            _cache[cache_key] = defaults
            return defaults

        # Try to match by name
        name_parts = pitcher_name.strip().split()
        if len(name_parts) >= 2:
            mask = stats["Name"].str.contains(name_parts[-1], case=False, na=False)
            match = stats[mask]
            if len(match) > 1:
                mask2 = match["Name"].str.contains(name_parts[0], case=False, na=False)
                match = match[mask2]
            if not match.empty:
                row = match.iloc[0]
                k_pct = row.get("K%", 0)
                if isinstance(k_pct, str):
                    k_pct = float(k_pct.strip("% ")) / 100
                result = {
                    "k_pct": round(float(k_pct), 4),
                    "k_per_9": round(float(row.get("K/9", 0)), 2),
                    "xk_pct": round(float(k_pct), 4),  # FG doesn't always have xK%
                }
                _cache[cache_key] = result
                return result
    except Exception:
        pass

    _cache[cache_key] = defaults
    return defaults


def get_pitcher_hand(df: pd.DataFrame) -> str:
    """Get pitcher throwing hand from Statcast data."""
    if df.empty or "p_throws" not in df.columns:
        return "R"
    vals = df["p_throws"].dropna()
    return str(vals.iloc[0]) if not vals.empty else "R"


def get_days_rest(game_logs: list[dict]) -> int:
    """Days since last start."""
    if not game_logs:
        return 5  # default assumption
    last_date = game_logs[-1]["date"]
    if hasattr(last_date, "date"):
        last_date = last_date.date()
    return (date.today() - last_date).days


def fetch_pitcher_stats(pitcher_id: int, pitcher_name: str) -> dict:
    """
    Main entry point. Returns a complete pitcher stats dict.
    """
    df = get_statcast_data(pitcher_id)
    game_logs = get_game_logs(df)
    fg = get_fangraphs_stats(pitcher_name)

    last_outing_pitches = game_logs[-1]["pitches"] if game_logs else 0

    return {
        "pitcher_id": pitcher_id,
        "pitcher_name": pitcher_name,
        "swstr_pct": calc_swstr_pct(df),
        "csw_pct": calc_csw_pct(df),
        "whiff_by_pitch": calc_whiff_by_pitch(df),
        "velocity": calc_velocity_trend(df),
        "k_pct": fg["k_pct"],
        "k_per_9": fg["k_per_9"],
        "xk_pct": fg["xk_pct"],
        "rolling_k_3": calc_rolling_ks(game_logs, 3),
        "rolling_k_5": calc_rolling_ks(game_logs, 5),
        "swstr_trend": calc_swstr_trend(df),
        "last_outing_pitches": last_outing_pitches,
        "days_rest": get_days_rest(game_logs),
        "pitcher_hand": get_pitcher_hand(df),
        "game_logs": game_logs,
    }


if __name__ == "__main__":
    # Test with a known pitcher — Corbin Burnes (MLB ID 669203)
    stats = fetch_pitcher_stats(669203, "Corbin Burnes")
    print("Corbin Burnes stats:")
    for k, v in stats.items():
        if k == "game_logs":
            print(f"  {k}: {len(v)} games")
        elif k == "whiff_by_pitch":
            print(f"  {k}:")
            for pt, pct in v.items():
                print(f"    {pt}: {pct:.1%}")
        else:
            print(f"  {k}: {v}")
