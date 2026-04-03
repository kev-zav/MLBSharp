"""
Fetch strikeout prop lines from The Odds API.
Conserves API requests — one call per run.
"""

import requests
from datetime import datetime
from config import ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT

_cache: dict[str, any] = {}

# Map Odds API full team names to abbreviations used by MLB Stats API
ODDS_TEAM_TO_ABBR = {
    "Arizona Diamondbacks": "AZ",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def fetch_strikeout_odds() -> tuple[list[dict], str | None]:
    """
    Pull pitcher strikeout props from all available US books.
    Returns (props_list, earliest_game_date_str).
    """
    if "odds" in _cache:
        return _cache["odds"], _cache.get("odds_date")

    if not ODDS_API_KEY or ODDS_API_KEY == "your_odds_api_key_here":
        print("  [ODDS] No API key set — skipping odds fetch.")
        print("  [ODDS] Get a free key at https://the-odds-api.com/")
        _cache["odds"] = []
        return [], None

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/events"
    try:
        resp = requests.get(url, params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "dateFormat": "iso",
        }, timeout=15)
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        print(f"  [ODDS] Error fetching events: {e}")
        _cache["odds"] = []
        return [], None

    if not events:
        print("  [ODDS] No MLB events found.")
        _cache["odds"] = []
        return [], None

    # Determine the earliest game date from events
    earliest_date = None
    for ev in events:
        ct = ev.get("commence_time", "")
        if ct:
            try:
                dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                d = dt.strftime("%Y-%m-%d")
                if earliest_date is None or d < earliest_date:
                    earliest_date = d
            except Exception:
                pass

    # Fetch pitcher strikeout props for each event
    all_props = []
    remaining = "?"
    for event in events:
        event_id = event.get("id")
        event_home = event.get("home_team", "")
        event_away = event.get("away_team", "")
        commence = event.get("commence_time", "")

        home_abbr = ODDS_TEAM_TO_ABBR.get(event_home, event_home)
        away_abbr = ODDS_TEAM_TO_ABBR.get(event_away, event_away)

        try:
            props_url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/events/{event_id}/odds"
            resp = requests.get(props_url, params={
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": "pitcher_strikeouts",
                "oddsFormat": "american",
            }, timeout=15)
            resp.raise_for_status()
            remaining = resp.headers.get("x-requests-remaining", remaining)
            data = resp.json()
        except Exception:
            continue

        bookmakers = data.get("bookmakers", [])
        for book in bookmakers:
            book_name = book.get("title", "Unknown")
            for market in book.get("markets", []):
                if market.get("key") != "pitcher_strikeouts":
                    continue
                for outcome in market.get("outcomes", []):
                    pitcher_name = outcome.get("description", "")
                    over_under = outcome.get("name", "")
                    line = outcome.get("point", 0)
                    odds = outcome.get("price", 0)

                    all_props.append({
                        "event_id": event_id,
                        "home_team": event_home,
                        "away_team": event_away,
                        "home_abbr": home_abbr,
                        "away_abbr": away_abbr,
                        "commence_time": commence,
                        "pitcher_name": pitcher_name,
                        "over_under": over_under,
                        "line": line,
                        "odds": odds,
                        "book": book_name,
                    })

    _cache["odds"] = all_props
    _cache["odds_date"] = earliest_date

    print(f"  [ODDS] Pulled {len(all_props)} props across {len(events)} events. "
          f"Games date: {earliest_date}. API remaining: {remaining}")

    return all_props, earliest_date


def get_pitcher_odds(all_props: list[dict], pitcher_name: str, team_abbr: str = "") -> dict:
    """
    Find the best odds for a specific pitcher's strikeout props.
    Matches by pitcher last name, with team abbreviation as tiebreaker.
    Returns {line, over_odds, under_odds, best_over, best_under, ladder}.
    Ladder contains all available K total lines for the pitcher.
    """
    from collections import Counter, defaultdict

    empty = {"line": 0, "over_odds": [], "under_odds": [], "best_over": None, "best_under": None, "ladder": []}
    if not all_props:
        return empty

    # Try exact name match first
    matching = [p for p in all_props if p["pitcher_name"].lower() == pitcher_name.lower()]

    # Fall back to last name match
    if not matching:
        last_name = pitcher_name.strip().split()[-1].lower()
        matching = [p for p in all_props if last_name in p["pitcher_name"].lower()]

    # If multiple matches, filter by team
    if len(matching) > 2 and team_abbr:
        team_filtered = [
            p for p in matching
            if p.get("home_abbr") == team_abbr or p.get("away_abbr") == team_abbr
        ]
        if team_filtered:
            matching = team_filtered

    if not matching:
        return empty

    # Group all props by line value
    lines_data = defaultdict(lambda: {"over": [], "under": []})
    for p in matching:
        if p["over_under"] == "Over":
            lines_data[p["line"]]["over"].append({"book": p["book"], "odds": p["odds"]})
        else:
            lines_data[p["line"]]["under"].append({"book": p["book"], "odds": p["odds"]})

    # Main line = most common across books
    line_counts = Counter(p["line"] for p in matching)
    main_line = line_counts.most_common(1)[0][0]

    # Build ladder — all available lines sorted ascending
    ladder = []
    for line_val in sorted(lines_data.keys()):
        over = sorted(lines_data[line_val]["over"], key=lambda x: x["odds"], reverse=True)
        under = sorted(lines_data[line_val]["under"], key=lambda x: x["odds"], reverse=True)
        ladder.append({
            "line": line_val,
            "over_odds": over,
            "under_odds": under,
            "best_over": over[0] if over else None,
            "best_under": under[0] if under else None,
        })

    main_over = sorted(lines_data[main_line]["over"], key=lambda x: x["odds"], reverse=True)
    main_under = sorted(lines_data[main_line]["under"], key=lambda x: x["odds"], reverse=True)

    return {
        "line": main_line,
        "over_odds": main_over,
        "under_odds": main_under,
        "best_over": main_over[0] if main_over else None,
        "best_under": main_under[0] if main_under else None,
        "ladder": ladder,
    }


if __name__ == "__main__":
    props, odds_date = fetch_strikeout_odds()
    if props:
        pitchers = sorted(set(p["pitcher_name"] for p in props))
        print(f"\nFound props for {len(pitchers)} pitchers (date: {odds_date}):")
        for name in pitchers:
            p_props = [p for p in props if p["pitcher_name"] == name]
            line = p_props[0]["line"]
            teams = f"{p_props[0]['away_abbr']} @ {p_props[0]['home_abbr']}"
            print(f"  {name}: {line} Ks — {teams}")
    else:
        print("No odds data available (check API key in .env)")
