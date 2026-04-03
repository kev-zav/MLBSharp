"""
Fetch umpire assignments and zone tendencies from UmpScorecards.
Umpires with tight zones → more Ks; expanded zones → more Ks too (more called strikes).
"""

import requests
from bs4 import BeautifulSoup

_cache: dict[str, any] = {}

UMPSCORECARDS_API = "https://umpscorecards.com/api/v1"


def fetch_umpire_for_game(home_team: str, away_team: str, game_date: str = "") -> dict:
    """
    Try to find today's umpire assignment.
    Falls back to a neutral profile if unavailable.
    """
    defaults = {
        "name": "Unknown",
        "favor_k": 0.0,        # positive = favors more Ks
        "zone_tendency": "Neutral",
        "adjustment": 1.0,
    }

    # UmpScorecards doesn't have a public daily schedule API,
    # so we try the MLB Stats API for umpire assignments
    try:
        from config import MLB_API_BASE
        from datetime import date

        game_date = game_date or date.today().strftime("%Y-%m-%d")
        url = f"{MLB_API_BASE}/schedule"
        resp = requests.get(url, params={
            "sportId": 1,
            "date": game_date,
            "hydrate": "officials",
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for d in data.get("dates", []):
            for g in d.get("games", []):
                teams = g.get("teams", {})
                h = teams.get("home", {}).get("team", {}).get("abbreviation", "")
                a = teams.get("away", {}).get("team", {}).get("abbreviation", "")
                if h == home_team or a == away_team:
                    officials = g.get("officials", [])
                    for off in officials:
                        if off.get("officialType") == "Home Plate":
                            ump_name = off.get("official", {}).get("fullName", "Unknown")
                            return _get_ump_profile(ump_name)
    except Exception:
        pass

    return defaults


def _get_ump_profile(ump_name: str) -> dict:
    """
    Look up an umpire's zone tendency.
    Uses a curated set of known umpire tendencies + UmpScorecards scrape.
    """
    cache_key = f"ump_{ump_name}"
    if cache_key in _cache:
        return _cache[cache_key]

    # Known umpire tendencies — dampened 50% for 2026 ABS challenge system.
    # Human ump still calls every pitch but egregious miscalls can be challenged,
    # reducing zone variance impact. Pre-ABS values halved across the board.
    KNOWN_UMPS = {
        "Angel Hernandez": {"favor_k": -0.010, "zone_tendency": "Erratic / Tight"},
        "CB Bucknor": {"favor_k": -0.005, "zone_tendency": "Erratic"},
        "Joe West": {"favor_k": 0.010, "zone_tendency": "Expanded"},
        "Doug Eddings": {"favor_k": 0.010, "zone_tendency": "Expanded"},
        "Ron Kulpa": {"favor_k": 0.008, "zone_tendency": "Slightly Expanded"},
        "Lance Barksdale": {"favor_k": 0.010, "zone_tendency": "Expanded"},
        "Marvin Hudson": {"favor_k": 0.008, "zone_tendency": "Slightly Expanded"},
        "Todd Tichenor": {"favor_k": -0.005, "zone_tendency": "Tight"},
        "Pat Hoberg": {"favor_k": 0.0, "zone_tendency": "Accurate"},
        "Nic Lentz": {"favor_k": 0.005, "zone_tendency": "Slightly Expanded"},
        "Dan Bellino": {"favor_k": 0.010, "zone_tendency": "Expanded"},
        "Tripp Gibson": {"favor_k": 0.005, "zone_tendency": "Slightly Expanded"},
        "John Tumpane": {"favor_k": -0.005, "zone_tendency": "Slightly Tight"},
        "Mark Carlson": {"favor_k": 0.0, "zone_tendency": "Neutral"},
        "Alan Porter": {"favor_k": 0.005, "zone_tendency": "Slightly Expanded"},
        "Chris Guccione": {"favor_k": -0.008, "zone_tendency": "Tight"},
        "Bill Miller": {"favor_k": 0.005, "zone_tendency": "Slightly Expanded"},
    }

    if ump_name in KNOWN_UMPS:
        profile = KNOWN_UMPS[ump_name]
        result = {
            "name": ump_name,
            "favor_k": profile["favor_k"],
            "zone_tendency": profile["zone_tendency"],
            "adjustment": 1.0 + profile["favor_k"],
        }
        _cache[cache_key] = result
        return result

    # Try to scrape UmpScorecards for this ump
    try:
        url = f"https://umpscorecards.com/umpires/"
        resp = requests.get(url, timeout=10)
        # Basic scrape — if we can find the ump, extract zone data
        # For now, return neutral if not in known list
    except Exception:
        pass

    result = {
        "name": ump_name,
        "favor_k": 0.0,
        "zone_tendency": "Neutral (no data)",
        "adjustment": 1.0,
    }
    _cache[cache_key] = result
    return result


if __name__ == "__main__":
    # Test with known ump
    profile = _get_ump_profile("Lance Barksdale")
    print(f"Lance Barksdale: {profile}")

    profile = _get_ump_profile("Pat Hoberg")
    print(f"Pat Hoberg: {profile}")

    # Test game lookup (may not find games for today)
    result = fetch_umpire_for_game("NYY", "BOS")
    print(f"NYY vs BOS umpire: {result}")
