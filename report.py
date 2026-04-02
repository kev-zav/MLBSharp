"""
Format and print the final MLB Strikeout Sharp Report.
"""

from datetime import date
from config import EDGE_STRONG, EDGE_MODERATE, EDGE_LEAN


def fmt_odds(odds: int) -> str:
    """Format American odds with +/- prefix."""
    return f"+{odds}" if odds > 0 else str(odds)


def fmt_pct(val: float) -> str:
    """Format a decimal as percentage."""
    return f"{val * 100:.1f}%"


def print_matchup(m: dict) -> str:
    """Format a single matchup entry."""
    lines = []
    pitcher = m.get("pitcher_name", "Unknown")
    opp_team = m.get("opp_team", "Unknown")
    play = m.get("play", "OVER")
    book_line = m.get("line", 0)

    lines.append(f"  {pitcher} vs {opp_team}  {play} {book_line}")
    lines.append(
        f"    Projected: {m.get('projected_ks', 0)} Ks | "
        f"Hit Rate: {m.get('hit_rate', 0):.0f}% | "
        f"Edge: {m.get('edge', 0):+.1f}pts"
    )
    lines.append(
        f"    SwStr%: {fmt_pct(m.get('swstr_pct', 0))} | "
        f"CSW%: {fmt_pct(m.get('csw_pct', 0))}"
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

    # Book odds
    over_odds = m.get("over_odds", [])
    under_odds = m.get("under_odds", [])
    if over_odds or under_odds:
        book_strs = []
        shown = set()
        for o in (over_odds[:3] if play == "OVER" else under_odds[:3]):
            if o["book"] not in shown:
                book_strs.append(f"{o['book']} {fmt_odds(o['odds'])}")
                shown.add(o["book"])
        if book_strs:
            lines.append(f"    Book: {' | '.join(book_strs)}")

    best = m.get("best_line")
    if best:
        lines.append(f"    → BEST LINE: {best['book']} {fmt_odds(best['odds'])}")

    # K probability distribution
    k_dist = m.get("k_distribution", {})
    if k_dist:
        # Only show K totals 4 through 10+, skip low values cluttering the display
        show_ks = [4, 5, 6, 7, 8, 9, "10+"]
        bar_parts = []
        for k in show_ks:
            pct = k_dist.get(k, 0)
            if pct >= 0.005:  # skip <0.5% totals
                bar_parts.append(f"{k}Ks:{pct*100:.0f}%")
        if bar_parts:
            lines.append(f"    K Dist: {' | '.join(bar_parts)}")

    return "\n".join(lines)


def print_parlay_suggestions(strong: list[dict], moderate: list[dict]) -> str:
    """Generate parlay suggestions from best plays."""
    lines = []
    candidates = strong + moderate

    if len(candidates) < 2:
        lines.append("  Not enough qualifying plays for parlays today.")
        return "\n".join(lines)

    # 2-leg parlay from top 2
    top2 = candidates[:2]
    lines.append(
        f"  2-leg: {top2[0]['pitcher_name']} {top2[0]['play']} {top2[0]['line']} + "
        f"{top2[1]['pitcher_name']} {top2[1]['play']} {top2[1]['line']}"
    )
    if top2[0].get("best_line") and top2[1].get("best_line"):
        lines.append(
            f"         Best books: {top2[0]['best_line']['book']}, "
            f"{top2[1]['best_line']['book']}"
        )

    # 3-leg parlay if we have 3+
    if len(candidates) >= 3:
        top3 = candidates[:3]
        names = " + ".join(
            f"{p['pitcher_name']} {p['play']} {p['line']}" for p in top3
        )
        lines.append(f"  3-leg: {names}")

    return "\n".join(lines)


def generate_report(scored_matchups: list[dict]) -> str:
    """Generate the full report string."""
    today = date.today().strftime("%B %d, %Y")

    # Sort by edge descending
    scored = sorted(scored_matchups, key=lambda x: x.get("edge", 0), reverse=True)

    strong = [m for m in scored if m.get("edge", 0) >= EDGE_STRONG]
    moderate = [m for m in scored if EDGE_MODERATE <= m.get("edge", 0) < EDGE_STRONG]
    lean = [m for m in scored if EDGE_LEAN <= m.get("edge", 0) < EDGE_MODERATE]
    no_value = [m for m in scored if m.get("edge", 0) < EDGE_LEAN]

    lines = []
    lines.append("=" * 50)
    lines.append("  MLB STRIKEOUT SHARP REPORT")
    lines.append(f"  {today}")
    lines.append("=" * 50)
    lines.append("")

    # Strong plays
    lines.append(f"STRONG PLAYS (edge > {EDGE_STRONG} pts)  [{len(strong)} plays]")
    lines.append("-" * 50)
    if strong:
        for m in strong:
            lines.append(print_matchup(m))
            lines.append("")
    else:
        lines.append("  No strong plays today.")
        lines.append("")

    # Moderate plays
    lines.append(f"MODERATE PLAYS (edge {EDGE_MODERATE}-{EDGE_STRONG} pts)  [{len(moderate)} plays]")
    lines.append("-" * 50)
    if moderate:
        for m in moderate:
            lines.append(print_matchup(m))
            lines.append("")
    else:
        lines.append("  No moderate plays today.")
        lines.append("")

    # Lean plays
    lines.append(f"LEAN / MONITOR (edge {EDGE_LEAN}-{EDGE_MODERATE} pts)  [{len(lean)} plays]")
    lines.append("-" * 50)
    if lean:
        for m in lean:
            lines.append(print_matchup(m))
            lines.append("")
    else:
        lines.append("  No lean plays today.")
        lines.append("")

    # No value
    lines.append(f"NO VALUE — PASS  [{len(no_value)} plays]")
    lines.append("-" * 50)
    if no_value:
        for m in no_value:
            lines.append(print_matchup(m))
            lines.append("")
    else:
        lines.append("  All matchups have some value today!")
        lines.append("")

    # Parlays
    lines.append("=" * 50)
    lines.append("PARLAY SUGGESTIONS")
    lines.append("=" * 50)
    lines.append(print_parlay_suggestions(strong, moderate))
    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with synthetic matchups
    test_matchups = [
        {
            "pitcher_name": "Corbin Burnes",
            "opp_team": "NYY",
            "play": "OVER",
            "line": 5.5,
            "projected_ks": 7.2,
            "hit_rate": 68,
            "edge": 12.3,
            "swstr_pct": 0.135,
            "csw_pct": 0.32,
            "pitcher_hand": "R",
            "opp_k_pct": 0.255,
            "opp_chase": 0.33,
            "umpire": {"name": "Lance Barksdale", "zone_tendency": "Expanded", "adjustment": 1.02},
            "weather": {"temp_f": 65, "wind_mph": 12, "description": "Partly Cloudy", "indoor": False},
            "over_odds": [{"book": "DraftKings", "odds": -120}, {"book": "FanDuel", "odds": -115}],
            "under_odds": [{"book": "BetMGM", "odds": -105}],
            "best_line": {"book": "FanDuel", "odds": -115},
        },
        {
            "pitcher_name": "Gerrit Cole",
            "opp_team": "BAL",
            "play": "OVER",
            "line": 6.5,
            "projected_ks": 7.8,
            "hit_rate": 62,
            "edge": 6.1,
            "swstr_pct": 0.128,
            "csw_pct": 0.30,
            "pitcher_hand": "R",
            "opp_k_pct": 0.238,
            "opp_chase": 0.29,
            "umpire": {"name": "Pat Hoberg", "zone_tendency": "Accurate", "adjustment": 1.0},
            "weather": {"temp_f": 72, "wind_mph": 5, "description": "Clear", "indoor": False},
            "over_odds": [{"book": "DraftKings", "odds": -110}],
            "under_odds": [],
            "best_line": {"book": "DraftKings", "odds": -110},
        },
    ]
    print(generate_report(test_matchups))
