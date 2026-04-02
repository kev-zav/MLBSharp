"""
Fetch today's probable starters and game info from the MLB Stats API.
Returns a list of matchup dicts with pitcher names, IDs, teams, venue, and game time.
"""

import requests
from datetime import date
from config import MLB_API_BASE


def get_todays_games(game_date: str | None = None) -> list[dict]:
    """
    Pull the MLB schedule for a given date (default: today).
    Returns raw game entries from the API.
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId": 1,
        "date": game_date,
        "hydrate": "probablePitcher,team,venue,linescore",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            games.append(g)
    return games


def parse_matchups(games: list[dict]) -> list[dict]:
    """
    Parse raw schedule data into clean matchup dicts.
    Each matchup has home/away pitcher info, team info, venue, and game time.
    """
    matchups = []

    for g in games:
        game_id = g.get("gamePk")
        game_time = g.get("gameDate", "")
        status = g.get("status", {}).get("detailedState", "")

        # Only include games that haven't started yet (or allow Final for backtesting)
        skip_statuses = {"In Progress", "Game Over", "Postponed", "Cancelled"}
        if status in skip_statuses:
            continue

        venue = g.get("venue", {})
        venue_name = venue.get("name", "Unknown")

        away = g.get("teams", {}).get("away", {})
        home = g.get("teams", {}).get("home", {})

        away_team = away.get("team", {})
        home_team = home.get("team", {})

        away_pitcher = away.get("probablePitcher", {})
        home_pitcher = home.get("probablePitcher", {})

        # Skip if no probable pitchers listed
        if not away_pitcher.get("id") or not home_pitcher.get("id"):
            continue

        matchup = {
            "game_id": game_id,
            "game_time": game_time,
            "venue": venue_name,
            "status": status,
            # Away pitcher faces home lineup
            "away_pitcher": {
                "id": away_pitcher["id"],
                "name": away_pitcher.get("fullName", "Unknown"),
                "team_id": away_team.get("id"),
                "team_name": away_team.get("name", "Unknown"),
                "team_abbr": away_team.get("abbreviation", ""),
            },
            # Home pitcher faces away lineup
            "home_pitcher": {
                "id": home_pitcher["id"],
                "name": home_pitcher.get("fullName", "Unknown"),
                "team_id": home_team.get("id"),
                "team_name": home_team.get("name", "Unknown"),
                "team_abbr": home_team.get("abbreviation", ""),
            },
            "away_team": {
                "id": away_team.get("id"),
                "team_name": away_team.get("name", "Unknown"),
                "team_abbr": away_team.get("abbreviation", ""),
            },
            "home_team": {
                "id": home_team.get("id"),
                "team_name": home_team.get("name", "Unknown"),
                "team_abbr": home_team.get("abbreviation", ""),
            },
        }
        matchups.append(matchup)

    return matchups


def fetch_starters(game_date: str | None = None) -> list[dict]:
    """Main entry point — returns parsed matchups for the day."""
    games = get_todays_games(game_date)
    return parse_matchups(games)


if __name__ == "__main__":
    matchups = fetch_starters()
    if not matchups:
        print("No games with probable starters found for today.")
    else:
        print(f"Found {len(matchups)} games with probable starters:\n")
        for m in matchups:
            ap = m["away_pitcher"]
            hp = m["home_pitcher"]
            print(f"  {ap['name']} ({ap['team_abbr']}) @ {hp['name']} ({hp['team_abbr']})")
            print(f"    Venue: {m['venue']}  |  Time: {m['game_time']}")
            print()
