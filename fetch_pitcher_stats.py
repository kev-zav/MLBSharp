"""
Fetch pitcher Statcast and FanGraphs stats via pybaseball.
Returns SwStr%, CSW%, K%, K/9, Whiff% by pitch, velocity trends,
rolling K averages, pitch count, and days rest.

K% source priority:
  1. FanGraphs (disk-cached, refreshed every 24h)
  2. Savant-derived K% from Statcast pitch data (fallback when FanGraphs is blocked)
"""

import json
import os
import warnings
from datetime import date, datetime, timedelta

import pandas as pd
from pybaseball import (
    playerid_lookup,
    statcast_pitcher,
    pitching_stats,
)
from config import (SEASON, LEAGUE_AVG_SWSTR, LEAGUE_AVG_K_PCT, LEAGUE_AVG_CSW,
                    LEAGUE_AVG_BB_PCT, LEAGUE_AVG_IP_PER_START, LEAGUE_AVG_PITCHES_PER_START)

warnings.filterwarnings("ignore")

# In-memory cache for Statcast data
_cache: dict[str, any] = {}

# Module-level FanGraphs DataFrame cache — fetched once per process, reused for all pitchers
_fg_df: pd.DataFrame | None = None


_FG_TABLE_CACHE_KEY = f"__fg_table_{SEASON}__"


def _get_fg_df() -> pd.DataFrame:
    """
    Fetch FanGraphs pitching stats once per run and cache in memory + disk.
    Disk cache ensures the full league table survives across GitHub Action runs,
    so new/rare pitchers are always covered without a fresh scrape.
    """
    global _fg_df
    if _fg_df is not None:
        return _fg_df

    # Check disk cache for the full table
    disk_cache = _load_fg_disk_cache()
    entry = disk_cache.get(_FG_TABLE_CACHE_KEY)
    if entry:
        age_hours = (datetime.now() - datetime.fromisoformat(entry["timestamp"])).total_seconds() / 3600
        if age_hours < _FG_CACHE_TTL_HOURS:
            try:
                _fg_df = pd.DataFrame(entry["data"])
                print(f"  [FG] Loaded {len(_fg_df)} pitchers from disk cache ({age_hours:.1f}h old).")
                return _fg_df
            except Exception:
                pass

    # Fetch fresh from FanGraphs
    try:
        print("  [FG] Pulling FanGraphs pitching stats (one-time fetch)...")
        _fg_df = pitching_stats(SEASON, SEASON, qual=1)
        print(f"  [FG] {len(_fg_df)} pitchers loaded from FanGraphs.")
        # Persist full table to disk
        disk_cache[_FG_TABLE_CACHE_KEY] = {
            "timestamp": datetime.now().isoformat(),
            "data": _fg_df.to_dict(orient="records"),
        }
        _save_fg_disk_cache(disk_cache)
    except Exception as e:
        print(f"  [FG] FanGraphs fetch failed: {e}")
        _fg_df = pd.DataFrame()
    return _fg_df

# Disk cache for FanGraphs data (avoids repeated scraping)
_FG_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fg_cache.json")
_FG_CACHE_TTL_HOURS = 48


def _load_fg_disk_cache() -> dict:
    try:
        with open(_FG_CACHE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_fg_disk_cache(cache: dict) -> None:
    try:
        with open(_FG_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _mlb_id_to_fangraphs_name(pitcher_name: str) -> tuple[str, str]:
    """Split 'First Last' into (last, first) for playerid_lookup."""
    parts = pitcher_name.strip().split()
    if len(parts) >= 2:
        return parts[-1], parts[0]
    return parts[0], ""


def get_statcast_data(pitcher_id: int, days_back: int = 60) -> pd.DataFrame:
    """Pull recent Statcast pitch-level data for a pitcher (regular season only)."""
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

    # Filter to regular season only (exclude spring training, exhibitions)
    if not df.empty and "game_type" in df.columns:
        df = df[df["game_type"] == "R"].copy()

    _cache[cache_key] = df
    return df


def calc_swstr_pct(df: pd.DataFrame) -> float:
    """Swinging strike rate from Statcast data."""
    if df.empty:
        return LEAGUE_AVG_SWSTR
    total = len(df)
    swinging_strikes = len(df[df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
        "foul_tip",
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


def calc_pitch_usage(df: pd.DataFrame) -> dict[str, float]:
    """Pitch usage % by pitch type."""
    if df.empty or "pitch_type" not in df.columns:
        return {}
    valid = df[df["pitch_type"].notna() & (df["pitch_type"] != "")]
    total = len(valid)
    if total == 0:
        return {}
    usage = {}
    for pt, grp in valid.groupby("pitch_type"):
        usage[str(pt)] = round(len(grp) / total, 4)
    return usage


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


def calc_k_pct_from_statcast(df: pd.DataFrame) -> float:
    """
    Derive K% directly from Statcast pitch data as a FanGraphs fallback.
    K% = strikeouts / total plate appearances.
    """
    if df.empty:
        return 0.0
    pa = df[df["events"].notna() & (df["events"] != "")]
    if len(pa) < 5:
        return 0.0
    ks = len(pa[pa["events"] == "strikeout"])
    return round(ks / len(pa), 4)


# Map MLB Stats API abbreviations to FanGraphs team names for disambiguation
_MLB_TO_FG_TEAM = {
    "AZ": "ARI", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CHW", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KC": "KCR", "LAA": "LAA", "LAD": "LAD",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY",
    "ATH": "OAK", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SD": "SDP",
    "SF": "SFG", "SEA": "SEA", "STL": "STL", "TB": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSN",
}


def get_prior_season_stats(pitcher_id: int, pitcher_name: str = "") -> dict:
    """
    Pull prior season K%, BB%, SwStr%, CSW%, IP/start, pitches/start from the
    local SQLite DB (built by build_pitcher_db.py). Falls back to a live
    Statcast pull if the DB has no entry, then to league averages.
    """
    prior_season = SEASON - 1
    cache_key = f"prior_stats_{pitcher_id}_{prior_season}"
    defaults = {
        "k_pct": LEAGUE_AVG_K_PCT,
        "bb_pct": LEAGUE_AVG_BB_PCT,
        "swstr_pct": LEAGUE_AVG_SWSTR,
        "csw_pct": LEAGUE_AVG_CSW,
        "ip_per_start": LEAGUE_AVG_IP_PER_START,
        "pitches_per_start": LEAGUE_AVG_PITCHES_PER_START,
        "whiff_by_pitch": {},
        "pitch_usage": {},
    }

    if cache_key in _cache:
        return _cache[cache_key]

    # 1. Check local SQLite DB first (fast, no API call)
    try:
        from build_pitcher_db import lookup
        row = lookup(pitcher_id, prior_season)
        if row:
            result = {
                "k_pct":              row.get("k_pct")              or LEAGUE_AVG_K_PCT,
                "bb_pct":             row.get("bb_pct")             or LEAGUE_AVG_BB_PCT,
                "swstr_pct":          row.get("swstr_pct")          or LEAGUE_AVG_SWSTR,
                "csw_pct":            row.get("csw_pct")            or LEAGUE_AVG_CSW,
                "ip_per_start":       row.get("ip_per_start")       or LEAGUE_AVG_IP_PER_START,
                "pitches_per_start":  row.get("pitches_per_start")  or LEAGUE_AVG_PITCHES_PER_START,
                "whiff_by_pitch":     row.get("whiff_by_pitch")     or {},
                "pitch_usage":        row.get("pitch_usage")        or {},
            }
            if pitcher_name:
                print(f"  [DB] {pitcher_name}: {prior_season} "
                      f"K%={result['k_pct']:.1%} BB%={result['bb_pct']:.1%} "
                      f"SwStr%={result['swstr_pct']:.1%} IP/GS={result['ip_per_start']:.1f}")
            _cache[cache_key] = result
            return result
    except Exception:
        pass

    # 2. Fall back to live Statcast pull (DB not built yet or pitcher missing)
    disk_cache = _load_fg_disk_cache()
    entry = disk_cache.get(cache_key)
    if entry and isinstance(entry.get("data"), dict):
        _cache[cache_key] = entry["data"]
        return entry["data"]

    try:
        df = statcast_pitcher(
            f"{prior_season}-03-01",
            f"{prior_season}-11-30",
            pitcher_id,
        )
        if not df.empty and "game_type" in df.columns:
            df = df[df["game_type"] == "R"]

        k_pct = calc_k_pct_from_statcast(df)
        swstr = calc_swstr_pct(df)
        csw = calc_csw_pct(df)

        # Derive BB% and pitches/start from Statcast
        bb_pct = LEAGUE_AVG_BB_PCT
        pitches_per_start = LEAGUE_AVG_PITCHES_PER_START
        if not df.empty:
            pa = df[df["events"].notna() & (df["events"] != "")]
            if len(pa) > 0:
                bb_pct = round(len(pa[pa["events"] == "walk"]) / len(pa), 4)
            import pandas as _pd
            df["game_date"] = _pd.to_datetime(df["game_date"])
            starts = df.groupby("game_date").size()
            if len(starts):
                pitches_per_start = round(starts.mean(), 1)

        result = {
            "k_pct":             k_pct  if k_pct  > 0 else LEAGUE_AVG_K_PCT,
            "bb_pct":            bb_pct,
            "swstr_pct":         swstr  if swstr  > 0 else LEAGUE_AVG_SWSTR,
            "csw_pct":           csw    if csw    > 0 else LEAGUE_AVG_CSW,
            "ip_per_start":      LEAGUE_AVG_IP_PER_START,
            "pitches_per_start": pitches_per_start,
        }

        if pitcher_name:
            print(f"  [PRIOR] {pitcher_name}: {prior_season} "
                  f"K%={result['k_pct']:.1%} BB%={result['bb_pct']:.1%} "
                  f"SwStr%={result['swstr_pct']:.1%} CSW%={result['csw_pct']:.1%}")
    except Exception:
        result = defaults

    disk_cache[cache_key] = {"timestamp": datetime.now().isoformat(), "data": result}
    _save_fg_disk_cache(disk_cache)
    _cache[cache_key] = result
    return result


def _regress_k_pct(k_pct: float, ip: float, pitcher_name: str = "", anchor: float = LEAGUE_AVG_K_PCT) -> float:
    """
    Regress K% toward an anchor weighted by innings pitched.
    Anchor defaults to league average but uses prior season K% when available.
    """
    regression_ip = 20.0
    regressed = (k_pct * ip + anchor * regression_ip) / (ip + regression_ip)
    if ip < 20 and abs(k_pct - anchor) > 0.05:
        print(f"  [REGRESS] {pitcher_name}: K% {k_pct:.1%} on {ip:.0f} IP "
              f"→ regressed to {regressed:.1%} (anchor: {anchor:.1%})")
    return round(regressed, 4)


def get_fangraphs_stats(pitcher_name: str, team_abbr: str = "", statcast_df: pd.DataFrame = None, pitcher_id: int = 0) -> dict:
    """
    Pull season-level K%, K/9 from FanGraphs via pybaseball.
    Uses disk cache to avoid repeated requests (refreshed every 24h).
    Falls back to Savant-derived K% when FanGraphs is blocked.
    """
    cache_key = f"{pitcher_name}_{SEASON}"
    defaults = {"k_pct": 0.0, "k_per_9": 0.0, "xk_pct": 0.0}

    # Check in-memory cache first
    mem_key = f"fg_{cache_key}"
    if mem_key in _cache:
        return _cache[mem_key]

    # Check disk cache
    disk_cache = _load_fg_disk_cache()
    entry = disk_cache.get(cache_key)
    if entry:
        age_hours = (datetime.now() - datetime.fromisoformat(entry["timestamp"])).total_seconds() / 3600
        if age_hours < _FG_CACHE_TTL_HOURS:
            _cache[mem_key] = entry["data"]
            return entry["data"]

    # Fetch prior season stats for regression anchor
    prior = get_prior_season_stats(pitcher_id, pitcher_name) if pitcher_id else {"k_pct": LEAGUE_AVG_K_PCT, "swstr_pct": LEAGUE_AVG_SWSTR, "csw_pct": LEAGUE_AVG_CSW}
    anchor = prior["k_pct"]

    # Try FanGraphs
    result = None
    try:
        stats = _get_fg_df()
        if not stats.empty:
            name_parts = pitcher_name.strip().split()
            if len(name_parts) >= 2:
                mask = stats["Name"].str.contains(name_parts[-1], case=False, na=False)
                match = stats[mask]

                if len(match) > 1:
                    mask2 = match["Name"].str.contains(name_parts[0], case=False, na=False)
                    if mask2.any():
                        match = match[mask2]

                if len(match) > 1 and team_abbr:
                    fg_team = _MLB_TO_FG_TEAM.get(team_abbr, team_abbr)
                    team_col = next((c for c in ["Team", "Tm", "team"] if c in match.columns), None)
                    if team_col:
                        team_mask = match[team_col].astype(str).str.contains(fg_team, case=False, na=False)
                        if team_mask.any():
                            match = match[team_mask]

                if len(match) > 1:
                    print(f"  [FG] {pitcher_name}: {len(match)} matches, using first")

                if not match.empty:
                    row = match.iloc[0]
                    k_pct = row.get("K%", 0)
                    if isinstance(k_pct, str):
                        k_pct = float(k_pct.strip("% ")) / 100
                    k_pct = float(k_pct)
                    ip = float(row.get("IP", 0) or 0)
                    regressed = _regress_k_pct(k_pct, ip, pitcher_name, anchor=anchor)
                    result = {
                        "k_pct": regressed,
                        "k_per_9": round(float(row.get("K/9", 0)), 2),
                        "xk_pct": regressed,
                        "k_pct_raw": round(k_pct, 4),
                        "ip": ip,
                        "source": "fangraphs",
                    }
    except Exception:
        pass

    # Fallback: derive K% from Statcast if FanGraphs failed
    if result is None and statcast_df is not None and not statcast_df.empty:
        sc_k_pct = calc_k_pct_from_statcast(statcast_df)
        if sc_k_pct > 0:
            pa = statcast_df[statcast_df["events"].notna() & (statcast_df["events"] != "")]
            ip_estimate = len(pa) / 3.0
            regressed = _regress_k_pct(sc_k_pct, ip_estimate, pitcher_name, anchor=anchor)
            print(f"  [SAVANT K%] {pitcher_name}: {sc_k_pct:.1%} from Statcast → regressed {regressed:.1%} (anchor: {anchor:.1%})")
            result = {
                "k_pct": regressed,
                "k_per_9": round(regressed * 27, 2),  # approximate K/9
                "xk_pct": regressed,
                "k_pct_raw": round(sc_k_pct, 4),
                "ip": round(ip_estimate, 1),
                "source": "savant",
            }

    if result is None:
        # Use prior season K% rather than falling back to 0.0 (which causes swstr*2.1 inflation)
        if anchor > 0:
            print(f"  [FG] {pitcher_name}: FanGraphs unavailable — using {anchor:.1%} prior-season K% as fallback")
            result = {
                "k_pct": anchor,
                "k_per_9": round(anchor * 27, 2),
                "xk_pct": anchor,
                "source": "prior_season",
            }
        else:
            result = defaults

    # Only cache to disk when we have real data (don't lock in failures for 24h)
    if result.get("k_pct", 0) > 0:
        disk_cache[cache_key] = {
            "timestamp": datetime.now().isoformat(),
            "data": result,
        }
        _save_fg_disk_cache(disk_cache)

    _cache[mem_key] = result
    return result


def get_pitcher_hand(df: pd.DataFrame) -> str:
    """Get pitcher throwing hand from Statcast data."""
    if df.empty or "p_throws" not in df.columns:
        return "R"
    vals = df["p_throws"].dropna()
    return str(vals.iloc[0]) if not vals.empty else "R"


def get_days_rest(game_logs: list[dict]) -> int:
    """Days since last start."""
    if not game_logs:
        return 5
    last_date = game_logs[-1]["date"]
    if hasattr(last_date, "date"):
        last_date = last_date.date()
    return (date.today() - last_date).days


# Reasonable bounds for pitcher stats
_STAT_BOUNDS = {
    "k_pct":    (0.05, 0.35),
    "xk_pct":   (0.05, 0.35),
    "swstr_pct": (0.03, 0.20),
    "csw_pct":  (0.10, 0.40),
    "rolling_k_3": (0.0, 13.0),
    "rolling_k_5": (0.0, 13.0),
}


def _sanitize(stats: dict) -> dict:
    """Clamp stats to reasonable bounds and print a warning if something is off."""
    flagged = []
    for key, (lo, hi) in _STAT_BOUNDS.items():
        val = stats.get(key)
        if val is None or val == 0:
            continue
        if val > hi:
            flagged.append(f"{key} {val:.3f} → capped {hi}")
            stats[key] = hi
        elif val < lo:
            flagged.append(f"{key} {val:.3f} → floored {lo}")
            stats[key] = lo
    if flagged:
        print(f"  [SANITY] {stats.get('pitcher_name','?')}: {' | '.join(flagged)}")
    return stats


def fetch_pitcher_stats(pitcher_id: int, pitcher_name: str, team_abbr: str = "") -> dict:
    """
    Main entry point. Returns a complete pitcher stats dict.
    """
    df = get_statcast_data(pitcher_id)
    game_logs = get_game_logs(df)
    fg = get_fangraphs_stats(pitcher_name, team_abbr, statcast_df=df, pitcher_id=pitcher_id)
    prior = get_prior_season_stats(pitcher_id, pitcher_name)  # returned from cache, no extra fetch

    num_starts = df["game_date"].nunique() if not df.empty and "game_date" in df.columns else 0

    # Blend current season SwStr%/CSW% with prior season based on starts.
    # Below 6 starts the sample is too noisy — weight heavily toward prior season.
    # At 6+ starts trust current season fully.
    if num_starts == 0:
        swstr_blended = prior["swstr_pct"]
        csw_blended = prior["csw_pct"]
    elif num_starts < 6:
        current_weight = num_starts / 6
        prior_weight = 1 - current_weight
        swstr_blended = calc_swstr_pct(df) * current_weight + prior["swstr_pct"] * prior_weight
        csw_blended = calc_csw_pct(df) * current_weight + prior["csw_pct"] * prior_weight
        print(f"  [SAMPLE] {pitcher_name}: {num_starts} starts — blending {current_weight:.0%} current / {prior_weight:.0%} prior SwStr%/CSW%")
    else:
        swstr_blended = calc_swstr_pct(df)
        csw_blended = calc_csw_pct(df)

    # Blend per-pitch whiff rates: prior season anchors, current season nudges
    # Same weight schedule as SwStr%/CSW% above
    prior_whiff = prior.get("whiff_by_pitch", {})
    prior_usage = prior.get("pitch_usage", {})
    current_whiff = calc_whiff_by_pitch(df)
    current_usage = calc_pitch_usage(df)

    if num_starts == 0:
        whiff_blended = prior_whiff
        usage_blended = prior_usage
    elif num_starts < 6:
        current_weight = num_starts / 6
        prior_weight = 1 - current_weight
        all_pts = set(prior_whiff) | set(current_whiff)
        whiff_blended = {}
        for pt in all_pts:
            p = prior_whiff.get(pt)
            c = current_whiff.get(pt)
            if p is not None and c is not None:
                whiff_blended[pt] = round(p * prior_weight + c * current_weight, 4)
            elif p is not None:
                whiff_blended[pt] = round(p, 4)
            elif c is not None:
                whiff_blended[pt] = round(c, 4)
        # Usage: blend if both exist, otherwise use whichever is available
        all_usage_pts = set(prior_usage) | set(current_usage)
        usage_blended = {}
        for pt in all_usage_pts:
            p = prior_usage.get(pt)
            c = current_usage.get(pt)
            if p is not None and c is not None:
                usage_blended[pt] = round(p * prior_weight + c * current_weight, 4)
            elif p is not None:
                usage_blended[pt] = round(p, 4)
            elif c is not None:
                usage_blended[pt] = round(c, 4)
    else:
        whiff_blended = current_whiff
        usage_blended = current_usage

    last_outing_pitches = game_logs[-1]["pitches"] if game_logs else 0

    stats = {
        "pitcher_id": pitcher_id,
        "pitcher_name": pitcher_name,
        "num_starts": num_starts,
        "swstr_pct": swstr_blended,
        "csw_pct": csw_blended,
        "whiff_by_pitch": whiff_blended,
        "pitch_usage": usage_blended,
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
        "k_pct_source": fg.get("source", "unknown"),
        "bb_pct": prior["bb_pct"],
        "ip_per_start": prior["ip_per_start"],
        "pitches_per_start": prior["pitches_per_start"],
    }

    return _sanitize(stats)


if __name__ == "__main__":
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
