"""
Fetch game-time weather from OpenWeatherMap API.
Temperature and wind affect strikeout rates (cold + wind = more Ks).
"""

import requests
from config import OPENWEATHER_API_KEY, VENUE_COORDS

# Domed / retractable roof stadiums where weather doesn't matter
INDOOR_VENUES = {
    "Tropicana Field",
    "Globe Life Field",
    "loanDepot park",
    "Minute Maid Park",
    "Rogers Centre",
    "Chase Field",
    "T-Mobile Park",       # retractable roof
    "American Family Field",  # retractable roof (MIL)
}


def fetch_weather(venue_name: str) -> dict:
    """
    Get current weather for a venue.
    Returns {temp_f, wind_mph, description, adjustment}.
    Adjustment is a multiplier: >1.0 = more Ks expected, <1.0 = fewer.
    """
    defaults = {
        "temp_f": 72,
        "wind_mph": 5,
        "description": "Unknown",
        "adjustment": 1.0,
        "indoor": False,
    }

    if venue_name in INDOOR_VENUES:
        return {
            "temp_f": 72,
            "wind_mph": 0,
            "description": "Dome / Retractable Roof",
            "adjustment": 1.0,
            "indoor": True,
        }

    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your_openweather_api_key_here":
        return defaults

    coords = VENUE_COORDS.get(venue_name)
    if not coords:
        return defaults

    lat, lon = coords
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        resp = requests.get(url, params={
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_API_KEY,
            "units": "imperial",
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return defaults

    temp_f = data.get("main", {}).get("temp", 72)
    wind_mph = data.get("wind", {}).get("speed", 5)
    desc = data.get("weather", [{}])[0].get("description", "Unknown")

    # Weather K adjustment:
    # Cold temps (< 60°F) increase Ks slightly
    # High wind (> 15 mph) increases Ks slightly
    adj = 1.0
    if temp_f < 50:
        adj += 0.03
    elif temp_f < 60:
        adj += 0.015
    elif temp_f > 90:
        adj -= 0.01  # extreme heat slightly hurts pitcher stamina

    if wind_mph > 20:
        adj += 0.02
    elif wind_mph > 15:
        adj += 0.01

    return {
        "temp_f": round(temp_f),
        "wind_mph": round(wind_mph),
        "description": desc.title(),
        "adjustment": round(adj, 3),
        "indoor": False,
    }


if __name__ == "__main__":
    for venue in ["Yankee Stadium", "Tropicana Field", "Wrigley Field"]:
        w = fetch_weather(venue)
        print(f"{venue}: {w['temp_f']}°F, {w['wind_mph']} mph wind, "
              f"\"{w['description']}\", adj={w['adjustment']}"
              f"{' (indoor)' if w['indoor'] else ''}")
