#!/usr/bin/env python3
"""
MLB Strikeout Sharp — Pitcher Database Seeder

Pulls prior season stats for every starting pitcher across all 30 MLB rosters
and stores them in a local SQLite database (pitchers.db).

Stats stored per pitcher per season:
  K%, BB%, SwStr%, CSW%, K/9, IP/start, pitches/start, total IP, GS

Source: FanGraphs (via pybaseball) — single batch pull, no per-pitcher API calls.

Usage:
  python3 build_pitcher_db.py              # seeds prior season for all 30 rosters
  python3 build_pitcher_db.py --season 2024
  python3 build_pitcher_db.py --team NYY   # single team
  python3 build_pitcher_db.py --refresh    # re-fetch even if already cached
"""

import argparse
import json
import os
import sqlite3
import sys
import warnings
from datetime import datetime, date

import requests

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "pitchers.db")
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

_MLB_TO_FG_TEAM = {
    "AZ": "ARI", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CHW", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KC": "KCR", "LAA": "LAA", "LAD": "LAD",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY",
    "ATH": "OAK", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SD": "SDP",
    "SF": "SFG", "SEA": "SEA", "STL": "STL", "TB": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSN",
}

# League average fallbacks
LEAGUE_AVG_K_PCT             = 0.224
LEAGUE_AVG_BB_PCT            = 0.082
LEAGUE_AVG_SWSTR             = 0.115
LEAGUE_AVG_CSW               = 0.270
LEAGUE_AVG_IP_PER_START      = 5.5
LEAGUE_AVG_PITCHES_PER_START = 88.0


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pitcher_seasons (
            pitcher_id          INTEGER NOT NULL,
            pitcher_name        TEXT    NOT NULL,
            team                TEXT,
            season              INTEGER NOT NULL,
            k_pct               REAL,
            bb_pct              REAL,
            swstr_pct           REAL,
            csw_pct             REAL,
            k_per_9             REAL,
            ip_per_start        REAL,
            pitches_per_start   REAL,
            total_ip            REAL,
            gs                  INTEGER,
            source              TEXT,
            updated_at          TEXT,
            whiff_by_pitch      TEXT,
            pitch_usage         TEXT,
            PRIMARY KEY (pitcher_id, season)
        )
    """)
    # Migrate existing DB: add columns if they don't exist yet
    for col, coltype in [("whiff_by_pitch", "TEXT"), ("pitch_usage", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE pitcher_seasons ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# MLB Stats API helpers
# ---------------------------------------------------------------------------

def get_all_teams() -> list[dict]:
    url = f"{MLB_API_BASE}/teams"
    resp = requests.get(url, params={"sportId": 1}, timeout=15)
    resp.raise_for_status()
    teams = []
    for t in resp.json().get("teams", []):
        teams.append({
            "id": t["id"],
            "name": t.get("name", ""),
            "abbr": t.get("abbreviation", ""),
        })
    return teams


def get_team_roster(team_id: int) -> list[dict]:
    url = f"{MLB_API_BASE}/teams/{team_id}/roster"
    resp = requests.get(url, params={"rosterType": "40Man"}, timeout=15)
    resp.raise_for_status()
    pitchers = []
    for p in resp.json().get("roster", []):
        pos = p.get("position", {}).get("abbreviation", "")
        if pos == "P":
            person = p.get("person", {})
            pitchers.append({
                "id": person["id"],
                "name": person.get("fullName", ""),
            })
    return pitchers


# ---------------------------------------------------------------------------
# FanGraphs batch fetch (all pitchers, one call)
# ---------------------------------------------------------------------------

def load_fangraphs_season(season: int):
    """
    Fetch the full FanGraphs pitching table for a season in one call.
    Returns a DataFrame indexed by (last_name_lower, first_initial_lower).
    """
    from pybaseball import pitching_stats
    df = pitching_stats(season, season, qual=1)
    return df


def _normalize(s: str) -> str:
    """Strip accents for fuzzy name matching (e.g. Vásquez → Vasquez)."""
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def lookup_fangraphs(df, pitcher_name: str, team_abbr: str) -> dict:
    """Match a pitcher in the pre-fetched FanGraphs DataFrame."""
    if df is None or df.empty:
        return {}

    # Normalize accents on both sides so Vásquez matches Vasquez
    norm_name = _normalize(pitcher_name.strip())
    parts = norm_name.split()
    norm_fg_names = df["Name"].apply(_normalize)

    mask = norm_fg_names.str.contains(parts[-1], case=False, na=False)
    match = df[mask]

    if len(match) > 1 and len(parts) >= 2:
        m2 = norm_fg_names[mask].str.contains(parts[0], case=False, na=False)
        if m2.any():
            match = match[m2]

    if len(match) > 1 and team_abbr:
        fg_team = _MLB_TO_FG_TEAM.get(team_abbr, team_abbr)
        team_col = next((c for c in ["Team", "Tm", "team"] if c in match.columns), None)
        if team_col:
            tm = match[team_col].astype(str).str.contains(fg_team, case=False, na=False)
            if tm.any():
                match = match[tm]

    if match.empty:
        return {}

    row = match.iloc[0]

    def pct(val):
        if isinstance(val, str):
            return float(val.strip("% ")) / 100
        return float(val or 0)

    gs  = int(row.get("GS", 0) or 0)
    k9  = float(row.get("K/9", 0) or 0)

    # Use Start-IP (innings only in starts) instead of total IP
    # This prevents inflated ip_per_start for pitchers who also relieved
    start_ip = float(row.get("Start-IP", 0) or 0)
    total_ip  = float(row.get("IP", 0) or 0)
    ip_for_starts = start_ip if start_ip > 0 else total_ip

    # Pitches in starts only: Starting column = games started as starter fraction
    # FanGraphs "Pitches" is total — scale by start ratio if available
    total_pitches = float(row.get("Pitches", 0) or 0)
    total_g = int(row.get("G", 0) or 0)
    if total_g > 0 and gs > 0:
        start_ratio = gs / total_g
        start_pitches = total_pitches * start_ratio
        pitches_per_start = round(start_pitches / gs, 1) if gs else LEAGUE_AVG_PITCHES_PER_START
    else:
        pitches_per_start = round(total_pitches / gs, 1) if gs else LEAGUE_AVG_PITCHES_PER_START

    swstr = pct(row.get("SwStr%", LEAGUE_AVG_SWSTR))
    csw   = pct(row.get("CSW%",   LEAGUE_AVG_CSW))

    return {
        "k_pct":              round(pct(row.get("K%", 0)), 4),
        "bb_pct":             round(pct(row.get("BB%", 0)), 4),
        "swstr_pct":          round(swstr, 4),
        "csw_pct":            round(csw, 4),
        "k_per_9":            round(k9, 2),
        "total_ip":           round(total_ip, 1),
        "gs":                 gs,
        "ip_per_start":       round(ip_for_starts / gs, 2) if gs else LEAGUE_AVG_IP_PER_START,
        "pitches_per_start":  pitches_per_start,
    }


# ---------------------------------------------------------------------------
# Statcast per-pitch whiff + usage (prior season)
# ---------------------------------------------------------------------------

_SWING_DESC = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
    "foul", "foul_bunt", "hit_into_play",
}
_MISS_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}


def fetch_pitch_type_stats(pitcher_id: int, season: int) -> dict:
    """
    Pull Statcast for a pitcher's prior season and compute:
      - whiff_by_pitch: {pitch_type → whiff%}  (min 20 swings per type)
      - pitch_usage:    {pitch_type → usage%}   (min 2% usage)

    Returns empty dicts if pull fails or pitcher threw <200 pitches (likely reliever).
    """
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from pybaseball import statcast_pitcher
    except ImportError:
        return {"whiff_by_pitch": {}, "pitch_usage": {}}

    try:
        df = statcast_pitcher(
            f"{season}-03-01",
            f"{season}-10-31",
            pitcher_id,
        )
    except Exception:
        return {"whiff_by_pitch": {}, "pitch_usage": {}}

    if df is None or df.empty or len(df) < 200:
        return {"whiff_by_pitch": {}, "pitch_usage": {}}

    # Regular season only
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    if len(df) < 200:
        return {"whiff_by_pitch": {}, "pitch_usage": {}}

    # Pitch usage
    valid = df[df["pitch_type"].notna() & (df["pitch_type"] != "")]
    total = len(valid)
    usage = {}
    for pt, grp in valid.groupby("pitch_type"):
        pct = len(grp) / total
        if pct >= 0.02:
            usage[str(pt)] = round(pct, 4)

    # Whiff by pitch type
    whiff = {}
    swings = df[df["description"].isin(_SWING_DESC)]
    for pt, grp in swings.groupby("pitch_type"):
        if str(pt) not in usage:
            continue
        if len(grp) < 20:
            continue
        misses = grp[grp["description"].isin(_MISS_DESC)]
        whiff[str(pt)] = round(len(misses) / len(grp), 4)

    return {"whiff_by_pitch": whiff, "pitch_usage": usage}


# ---------------------------------------------------------------------------
# Core seeder
# ---------------------------------------------------------------------------

def seed_pitcher(pitcher_id: int, pitcher_name: str, team_abbr: str,
                 season: int, fg_df, refresh: bool = False,
                 fetch_pitch_types: bool = True) -> bool:
    conn = get_conn()
    existing = conn.execute(
        "SELECT whiff_by_pitch FROM pitcher_seasons WHERE pitcher_id=? AND season=?",
        (pitcher_id, season)
    ).fetchone()

    # Skip entirely if already seeded with pitch-type data (unless --refresh)
    if not refresh and existing and existing[0]:
        conn.close()
        return False
    # Re-seed FG stats if missing, but only fetch Statcast pitch data if column is empty
    needs_pitch_data = existing is None or not existing[0]

    fg = lookup_fangraphs(fg_df, pitcher_name, team_abbr)

    k_pct             = fg.get("k_pct")            or LEAGUE_AVG_K_PCT
    bb_pct            = fg.get("bb_pct")           or LEAGUE_AVG_BB_PCT
    swstr_pct         = fg.get("swstr_pct")        or LEAGUE_AVG_SWSTR
    csw_pct           = fg.get("csw_pct")          or LEAGUE_AVG_CSW
    k_per_9           = fg.get("k_per_9")          or round(k_pct * 27, 2)
    total_ip          = fg.get("total_ip")         or 0.0
    gs                = fg.get("gs")               or 0
    ip_per_start      = fg.get("ip_per_start")     or LEAGUE_AVG_IP_PER_START
    pitches_per_start = fg.get("pitches_per_start") or LEAGUE_AVG_PITCHES_PER_START
    source            = "fangraphs" if fg else "league_avg"

    # Fetch per-pitch Statcast data for this season
    whiff_json = None
    usage_json = None
    if fetch_pitch_types and (needs_pitch_data or refresh):
        pt_stats = fetch_pitch_type_stats(pitcher_id, season)
        if pt_stats["whiff_by_pitch"]:
            whiff_json = json.dumps(pt_stats["whiff_by_pitch"])
            usage_json = json.dumps(pt_stats["pitch_usage"])

    conn.execute("""
        INSERT OR REPLACE INTO pitcher_seasons
            (pitcher_id, pitcher_name, team, season, k_pct, bb_pct, swstr_pct,
             csw_pct, k_per_9, ip_per_start, pitches_per_start, total_ip, gs,
             source, updated_at, whiff_by_pitch, pitch_usage)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        pitcher_id, pitcher_name, team_abbr, season,
        k_pct, bb_pct, swstr_pct, csw_pct, k_per_9,
        ip_per_start, pitches_per_start, total_ip, gs,
        source, datetime.now().isoformat(),
        whiff_json, usage_json,
    ))
    conn.commit()
    conn.close()
    return True


def lookup(pitcher_id: int, season: int) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM pitcher_seasons WHERE pitcher_id=? AND season=?",
        (pitcher_id, season)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    # Parse JSON columns back to dicts
    for col in ("whiff_by_pitch", "pitch_usage"):
        raw = d.get(col)
        if raw:
            try:
                d[col] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                d[col] = {}
        else:
            d[col] = {}
    return d


def backfill_pitch_types(pitcher_id: int, pitcher_name: str, season: int) -> bool:
    """
    Only update whiff_by_pitch and pitch_usage for an existing DB row.
    Skips if already populated. Does NOT touch k_pct or other FG stats.
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT whiff_by_pitch FROM pitcher_seasons WHERE pitcher_id=? AND season=?",
        (pitcher_id, season)
    ).fetchone()
    conn.close()

    if not row:
        return False  # pitcher not in DB at all — skip
    if row[0]:
        return False  # already has pitch-type data

    pt_stats = fetch_pitch_type_stats(pitcher_id, season)
    if not pt_stats["whiff_by_pitch"]:
        return False  # reliever or no data

    conn = get_conn()
    conn.execute(
        "UPDATE pitcher_seasons SET whiff_by_pitch=?, pitch_usage=?, updated_at=? "
        "WHERE pitcher_id=? AND season=?",
        (
            json.dumps(pt_stats["whiff_by_pitch"]),
            json.dumps(pt_stats["pitch_usage"]),
            datetime.now().isoformat(),
            pitcher_id, season,
        )
    )
    conn.commit()
    conn.close()
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Seed pitcher prior-season DB")
    parser.add_argument("--season", type=int, default=None,
                        help="Season to seed (default: current year - 1)")
    parser.add_argument("--team", type=str, default=None,
                        help="Seed only this team abbreviation (e.g. NYY)")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch even if already in DB")
    parser.add_argument("--no-pitch-types", action="store_true",
                        help="Skip Statcast per-pitch pull (faster, no whiff_by_pitch data)")
    parser.add_argument("--pitch-types-only", action="store_true",
                        help="Only backfill whiff_by_pitch/pitch_usage for existing rows (skip FG stats)")
    args = parser.parse_args()

    season = args.season or (date.today().year - 1)

    print(f"\n{'='*55}")
    print(f"  MLB Strikeout Sharp — Pitcher DB Seeder")
    print(f"  Season: {season}  |  DB: {DB_PATH}")
    print(f"{'='*55}\n")

    init_db()

    print("[1/4] Fetching team list from MLB API...")
    teams = get_all_teams()
    if args.team:
        teams = [t for t in teams if t["abbr"].upper() == args.team.upper()]
        if not teams:
            print(f"  Team '{args.team}' not found.")
            sys.exit(1)
    print(f"  {len(teams)} teams\n")

    print("[2/4] Fetching rosters...")
    all_pitchers: list[dict] = []
    for team in teams:
        try:
            pitchers = get_team_roster(team["id"])
            for p in pitchers:
                p["team_abbr"] = team["abbr"]
            all_pitchers.extend(pitchers)
        except Exception as e:
            print(f"  Roster fetch failed for {team['abbr']}: {e}")

    seen = {}
    for p in all_pitchers:
        if p["id"] not in seen:
            seen[p["id"]] = p
    unique_pitchers = list(seen.values())
    print(f"  {len(unique_pitchers)} unique pitchers across {len(teams)} teams\n")

    print(f"[3/4] Loading FanGraphs {season} stats (one batch fetch)...")
    try:
        fg_df = load_fangraphs_season(season)
        print(f"  {len(fg_df)} pitchers loaded from FanGraphs\n")
    except Exception as e:
        print(f"  FanGraphs fetch failed: {e}")
        fg_df = None

    print(f"[4/4] Seeding database...\n")
    seeded = 0
    skipped = 0
    errors = 0

    for i, p in enumerate(unique_pitchers, 1):
        pid  = p["id"]
        name = p["name"]
        abbr = p["team_abbr"]
        print(f"  [{i:3d}/{len(unique_pitchers)}] {name} ({abbr})...", end="", flush=True)
        try:
            if args.pitch_types_only:
                wrote = backfill_pitch_types(pid, name, season)
            else:
                wrote = seed_pitcher(pid, name, abbr, season, fg_df,
                                     refresh=args.refresh,
                                     fetch_pitch_types=not args.no_pitch_types)
            if wrote:
                row = lookup(pid, season)
                pitch_types = list(row.get("whiff_by_pitch", {}).keys())
                pt_str = f" | pitches: {','.join(pitch_types)}" if pitch_types else ""
                print(f" K%={row['k_pct']:.1%} BB%={row['bb_pct']:.1%} "
                      f"SwStr%={row['swstr_pct']:.1%} IP/GS={row['ip_per_start']:.1f} "
                      f"P/GS={row['pitches_per_start']:.0f} [{row['source']}]{pt_str}")
                seeded += 1
            else:
                print(" (already in DB, skipped)")
                skipped += 1
        except Exception as e:
            print(f" ERROR: {e}")
            errors += 1

    print(f"\n{'='*55}")
    print(f"  Done. Seeded: {seeded}  |  Skipped: {skipped}  |  Errors: {errors}")
    print(f"  DB: {DB_PATH}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
