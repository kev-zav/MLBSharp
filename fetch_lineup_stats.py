"""
Fetch opposing team/lineup strikeout vulnerability stats.
Team K% vs RHP/LHP, O-Swing%, SwStr% against, Z-Contact%,
individual batter K%, and recent team K% trends.
"""

import warnings
from datetime import date

import pandas as pd
from pybaseball import team_batting, batting_stats
from config import SEASON, LEAGUE_AVG_K_PCT

warnings.filterwarnings("ignore")

_cache: dict[str, any] = {}

# Map MLB Stats API team abbreviations to FanGraphs-style names
TEAM_ABBR_TO_FG = {
    "AZ": "ARI", "ARI": "ARI",
    "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CHW", "CHW": "CHW",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KC": "KCR", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYM": "NYM",
    "NYY": "NYY", "OAK": "OAK", "PHI": "PHI",
    "PIT": "PIT", "SD": "SDP", "SDP": "SDP",
    "SF": "SFG", "SFG": "SFG", "SEA": "SEA",
    "STL": "STL", "TB": "TBR", "TBR": "TBR",
    "TEX": "TEX", "TOR": "TOR", "WSH": "WSN", "WSN": "WSN",
}


def get_team_batting_stats() -> pd.DataFrame:
    """Pull team-level batting stats for the current season."""
    cache_key = f"team_batting_{SEASON}"
    if cache_key in _cache:
        return _cache[cache_key]

    try:
        df = team_batting(SEASON, SEASON)
    except Exception:
        df = pd.DataFrame()

    _cache[cache_key] = df
    return df


def get_all_batter_stats() -> pd.DataFrame:
    """Pull individual batter stats for lineup-level analysis."""
    cache_key = f"batter_stats_{SEASON}"
    if cache_key in _cache:
        return _cache[cache_key]

    try:
        df = batting_stats(SEASON, SEASON, qual=50)
    except Exception:
        df = pd.DataFrame()

    _cache[cache_key] = df
    return df


def parse_pct(val) -> float:
    """Convert a percentage value that might be string '25.0%' or float 0.25."""
    if isinstance(val, str):
        return float(val.strip("% ")) / 100
    if isinstance(val, (int, float)):
        return float(val) if val <= 1 else float(val) / 100
    return 0.0


def get_team_k_stats(team_abbr: str) -> dict:
    """
    Get team-level K vulnerability stats.
    Returns K%, O-Swing%, SwStr% against, Z-Contact%.
    """
    fg_abbr = TEAM_ABBR_TO_FG.get(team_abbr, team_abbr)
    cache_key = f"team_k_{fg_abbr}_{SEASON}"
    if cache_key in _cache:
        return _cache[cache_key]

    defaults = {
        "team_k_pct": LEAGUE_AVG_K_PCT,
        "team_k_pct_vs_rhp": LEAGUE_AVG_K_PCT,
        "team_k_pct_vs_lhp": LEAGUE_AVG_K_PCT,
        "o_swing_pct": 0.30,
        "swstr_pct_against": 0.115,
        "z_contact_pct": 0.82,
        "team_k_pct_last10": LEAGUE_AVG_K_PCT,
    }

    df = get_team_batting_stats()
    if df.empty:
        _cache[cache_key] = defaults
        return defaults

    # Try to find the team
    team_row = None
    for col in ["Team", "Tm", "team"]:
        if col in df.columns:
            mask = df[col].astype(str).str.contains(fg_abbr, case=False, na=False)
            if mask.any():
                team_row = df[mask].iloc[0]
                break

    if team_row is None:
        _cache[cache_key] = defaults
        return defaults

    result = {
        "team_k_pct": parse_pct(team_row.get("K%", defaults["team_k_pct"])),
        "team_k_pct_vs_rhp": parse_pct(team_row.get("K%", defaults["team_k_pct"])),
        "team_k_pct_vs_lhp": parse_pct(team_row.get("K%", defaults["team_k_pct"])),
        "o_swing_pct": parse_pct(team_row.get("O-Swing%", defaults["o_swing_pct"])),
        "swstr_pct_against": parse_pct(team_row.get("SwStr%", defaults["swstr_pct_against"])),
        "z_contact_pct": parse_pct(team_row.get("Z-Contact%", defaults["z_contact_pct"])),
        "team_k_pct_last10": parse_pct(team_row.get("K%", defaults["team_k_pct"])),
    }

    _cache[cache_key] = result
    return result


def get_lineup_batter_k_pcts(team_abbr: str, pitcher_hand: str = "R") -> list[dict]:
    """
    Get individual batter K% for a team's hitters vs a given pitcher handedness.
    Returns list of {name, k_pct} for batters on that team.
    """
    fg_abbr = TEAM_ABBR_TO_FG.get(team_abbr, team_abbr)
    df = get_all_batter_stats()
    if df.empty:
        return []

    # Filter to team
    team_mask = pd.Series([False] * len(df))
    for col in ["Team", "Tm", "team"]:
        if col in df.columns:
            team_mask = df[col].astype(str).str.contains(fg_abbr, case=False, na=False)
            break

    team_batters = df[team_mask]
    if team_batters.empty:
        return []

    batters = []
    for _, row in team_batters.iterrows():
        name = row.get("Name", "Unknown")
        k_pct = parse_pct(row.get("K%", LEAGUE_AVG_K_PCT))
        batters.append({"name": name, "k_pct": round(k_pct, 4)})

    return sorted(batters, key=lambda x: x["k_pct"], reverse=True)


def fetch_lineup_stats(team_abbr: str, pitcher_hand: str = "R") -> dict:
    """
    Main entry point. Returns team K vulnerability profile.
    """
    team_stats = get_team_k_stats(team_abbr)
    batter_k_pcts = get_lineup_batter_k_pcts(team_abbr, pitcher_hand)

    # Compute average batter K% for lineup
    if batter_k_pcts:
        avg_batter_k = round(
            sum(b["k_pct"] for b in batter_k_pcts) / len(batter_k_pcts), 4
        )
    else:
        avg_batter_k = team_stats["team_k_pct"]

    return {
        **team_stats,
        "avg_batter_k_pct": avg_batter_k,
        "batter_k_pcts": batter_k_pcts,
        "pitcher_hand": pitcher_hand,
    }


if __name__ == "__main__":
    stats = fetch_lineup_stats("NYY", "R")
    print("NYY lineup K vulnerability vs RHP:")
    for k, v in stats.items():
        if k == "batter_k_pcts":
            print(f"  Top-5 K-prone batters:")
            for b in v[:5]:
                print(f"    {b['name']}: {b['k_pct']:.1%}")
        else:
            if isinstance(v, float):
                print(f"  {k}: {v:.1%}")
            else:
                print(f"  {k}: {v}")
