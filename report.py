"""
Format and print the final MLB Strikeout Sharp Report.
"""

from datetime import date


def fmt_pct(val: float) -> str:
    """Format a decimal as percentage."""
    return f"{val * 100:.1f}%"


def print_matchup(m: dict) -> str:
    """Format a single matchup entry."""
    lines = []
    pitcher = m.get("pitcher_name", "Unknown")
    opp_team = m.get("opp_team", "Unknown")
    proj = m.get("projected_ks", 0)
    lineup_confirmed = m.get("lineup_confirmed", False)
    lineup_tag = " [lineup confirmed]" if lineup_confirmed else ""

    lines.append(f"  {pitcher} vs {opp_team}  →  {proj} Ks projected{lineup_tag}")
    lines.append(
        f"    SwStr%: {fmt_pct(m.get('swstr_pct', 0))} | "
        f"CSW%: {fmt_pct(m.get('csw_pct', 0))} | "
        f"K%: {fmt_pct(m.get('k_pct', 0))}"
    )
    lines.append(
        f"    Opp K% vs {m.get('pitcher_hand', 'R')}HP: "
        f"{fmt_pct(m.get('opp_k_pct', 0))} | "
        f"Chase Rate: {fmt_pct(m.get('opp_chase', 0))}"
    )

    ump = m.get("umpire", {})
    lines.append(
        f"    Ump: {ump.get('name', 'TBD')} — "
        f"{ump.get('zone_tendency', 'N/A')} "
        f"({ump.get('adjustment', 1.0):.3f}x)"
    )

    weather = m.get("weather", {})
    if weather.get("indoor"):
        lines.append(f"    Weather: Dome (no impact)")
    else:
        lines.append(
            f"    Weather: {weather.get('temp_f', '?')}°F, "
            f"{weather.get('wind_mph', '?')} mph — "
            f"\"{weather.get('description', 'N/A')}\""
        )

    # K probability distribution
    k_dist = m.get("k_distribution", {})
    if k_dist:
        show_ks = [4, 5, 6, 7, 8, 9, "10+"]
        bar_parts = []
        for k in show_ks:
            pct = k_dist.get(k, 0)
            if pct >= 0.005:
                bar_parts.append(f"{k}Ks:{pct*100:.0f}%")
        if bar_parts:
            lines.append(f"    K Dist: {' | '.join(bar_parts)}")

    return "\n".join(lines)


def generate_report(scored_matchups: list[dict]) -> str:
    """Generate the full report string, sorted by projected Ks descending."""
    today = date.today().strftime("%B %d, %Y")
    scored = sorted(scored_matchups, key=lambda x: x.get("projected_ks", 0), reverse=True)

    lines = []
    lines.append("=" * 50)
    lines.append("  MLB STRIKEOUT SHARP — K PROJECTIONS")
    lines.append(f"  {today}")
    lines.append("=" * 50)
    lines.append("")

    for m in scored:
        lines.append(print_matchup(m))
        lines.append("")

    lines.append("=" * 50)
    lines.append(f"  {len(scored)} pitchers projected")
    lines.append("=" * 50)

    return "\n".join(lines)


if __name__ == "__main__":
    test_matchups = [
        {
            "pitcher_name": "Corbin Burnes",
            "opp_team": "NYY",
            "projected_ks": 7.2,
            "swstr_pct": 0.135,
            "csw_pct": 0.32,
            "k_pct": 0.28,
            "pitcher_hand": "R",
            "opp_k_pct": 0.255,
            "opp_chase": 0.33,
            "lineup_confirmed": True,
            "umpire": {"name": "Lance Barksdale", "zone_tendency": "Expanded", "adjustment": 1.02},
            "weather": {"temp_f": 65, "wind_mph": 12, "description": "Partly Cloudy", "indoor": False},
            "k_distribution": {4: 0.08, 5: 0.14, 6: 0.20, 7: 0.22, 8: 0.18, 9: 0.11, "10+": 0.07},
        },
    ]
    print(generate_report(test_matchups))
