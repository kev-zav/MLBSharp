"""
Projection engine for strikeout props.
Combines pitcher stats, lineup vulnerability, park factor, weather, and umpire
to project a K total and compare against the book line.
"""

from config import (
    LEAGUE_AVG_K_PCT,
    LEAGUE_AVG_SWSTR,
    DEFAULT_BATTERS_FACED,
    DEFAULT_PITCH_LIMIT,
    PARK_FACTOR_DEFAULT,
)


# League average whiff rates by pitch type (Statcast 2023-2025 baselines)
_LEAGUE_WHIFF_BY_PITCH = {
    "FF": 0.22, "SI": 0.17, "FC": 0.23,
    "SL": 0.34, "ST": 0.38, "CU": 0.28, "KC": 0.27,
    "CH": 0.32, "FS": 0.37, "FO": 0.32, "CS": 0.26,
}
_DEFAULT_PITCH_WHIFF = 0.27


def calc_pitch_mix_adjustment(pitcher_stats: dict, lineup_stats: dict) -> float:
    """
    Multiplier based on how well pitcher's arsenal matches lineup vulnerabilities.
    Cross-references pitcher pitch usage + whiff rate vs team whiff rate by pitch type.
    >1.0 = lineup is weak against this pitcher's stuff. <1.0 = lineup handles it well.
    Clamped to ±8% impact.
    """
    pitch_usage = pitcher_stats.get("pitch_usage", {})
    team_whiff_by_pitch = lineup_stats.get("whiff_by_pitch_type", {})

    if not pitch_usage or not team_whiff_by_pitch:
        return 1.0

    weighted_diff = 0.0
    total_weight = 0.0

    for pitch_type, usage in pitch_usage.items():
        if usage < 0.05:  # ignore pitches thrown < 5% of the time
            continue
        league_avg = _LEAGUE_WHIFF_BY_PITCH.get(pitch_type, _DEFAULT_PITCH_WHIFF)
        team_whiff = team_whiff_by_pitch.get(pitch_type, league_avg)
        weighted_diff += usage * (team_whiff - league_avg)
        total_weight += usage

    if total_weight == 0:
        return 1.0

    # Normalize and scale conservatively
    raw_adj = weighted_diff / total_weight
    multiplier = 1.0 + (raw_adj / LEAGUE_AVG_K_PCT) * 0.4  # dampened 40%
    return max(0.92, min(1.08, multiplier))


def estimate_batters_faced(pitcher_stats: dict) -> float:
    """
    Estimate batters faced based on pitch limit, days rest, and recent workload.
    Uses pitcher's historical pitches/start and IP/start from the DB when available.
    """
    days_rest = pitcher_stats.get("days_rest", 5)

    # Use pitcher's historical pitch limit from DB, fall back to league default
    # Cap at 120 to guard against bad DB entries (e.g. relievers with 2 GS inflating totals)
    raw_pitches = pitcher_stats.get("pitches_per_start") or DEFAULT_PITCH_LIMIT
    pitch_limit = raw_pitches if raw_pitches <= 120 else DEFAULT_PITCH_LIMIT

    # Short rest = lower pitch limit
    if days_rest <= 3:
        pitch_limit = min(pitch_limit, 75)
    elif days_rest == 4:
        pitch_limit = min(pitch_limit, 85)

    last_pitches = pitcher_stats.get("last_outing_pitches", 0)
    if last_pitches > 105:
        pitch_limit = min(pitch_limit, pitch_limit * 0.90)  # managed workload

    # Adjust pitches per batter for walk rate — more walks = more pitches per PA
    # League avg BB% ~8.2% ≈ 3.89 pitches/BF; each 1% above adds ~0.05 pitches/BF
    bb_pct = pitcher_stats.get("bb_pct", 0.082)
    pitches_per_bf = 3.89 + (bb_pct - 0.082) * 5.0
    pitches_per_bf = max(3.60, min(4.20, pitches_per_bf))  # clamp to reasonable range

    bf_from_pitches = pitch_limit / pitches_per_bf

    # Blend with IP/start-based estimate when available (4.35 BF per inning avg)
    # Cap at 7.5 IP to guard against bad DB entries (relievers/openers with inflated totals)
    raw_ip = pitcher_stats.get("ip_per_start")
    ip_per_start = raw_ip if raw_ip and raw_ip <= 7.5 else None
    if ip_per_start and ip_per_start > 0:
        bf_from_ip = ip_per_start * 4.35
        bf = bf_from_pitches * 0.5 + bf_from_ip * 0.5
    else:
        bf = bf_from_pitches

    return round(bf, 1)


def calc_pitcher_xk_rate(pitcher_stats: dict) -> float:
    """
    Expected K rate per batter faced, weighted across metrics.
    """
    swstr = pitcher_stats.get("swstr_pct", LEAGUE_AVG_SWSTR)
    csw = pitcher_stats.get("csw_pct", 0.0)
    k_pct = pitcher_stats.get("k_pct", 0.0)
    xk_pct = pitcher_stats.get("xk_pct", 0.0)

    # If we have FanGraphs K%, use it as the base
    if k_pct > 0:
        base_rate = k_pct
    elif xk_pct > 0:
        base_rate = xk_pct
    else:
        # Estimate from SwStr% — roughly K% ≈ SwStr% * 2.1 (empirical)
        base_rate = swstr * 2.1

    # Blend with CSW if available (CSW correlates ~0.7 with K%)
    if csw > 0:
        csw_implied_k = csw * 0.75  # CSW → K% conversion factor
        base_rate = base_rate * 0.65 + csw_implied_k * 0.35

    # Recent form adjustment from SwStr% trend
    swstr_trend = pitcher_stats.get("swstr_trend", 0.0)
    trend_adj = swstr_trend * 1.5  # SwStr trend amplified to K impact
    base_rate += trend_adj

    # Velocity trend adjustment
    velo = pitcher_stats.get("velocity", {})
    velo_trend = velo.get("trend", 0.0)
    if velo_trend > 1.0:
        base_rate *= 1.02  # velo up = more Ks
    elif velo_trend < -1.0:
        base_rate *= 0.98  # velo down = fewer Ks

    return max(0.05, min(0.45, base_rate))  # clamp between 5% and 45%


def calc_lineup_adjustment(lineup_stats: dict) -> float:
    """
    How much this lineup's K vulnerability differs from league average.
    Returns a multiplier (>1.0 = more K-prone, <1.0 = less K-prone).
    """
    team_k = lineup_stats.get("team_k_pct", LEAGUE_AVG_K_PCT)
    chase = lineup_stats.get("o_swing_pct", 0.30)
    swstr_against = lineup_stats.get("swstr_pct_against", LEAGUE_AVG_SWSTR)
    z_contact = lineup_stats.get("z_contact_pct", 0.82)
    bb_pct = lineup_stats.get("bb_pct", 0.085)

    # K% differential vs league average
    k_diff = team_k - LEAGUE_AVG_K_PCT

    # Chase rate premium (league avg ~30%)
    chase_diff = chase - 0.30

    # SwStr% against differential
    swstr_diff = swstr_against - LEAGUE_AVG_SWSTR

    # Z-Contact% — lower = worse contact in zone = more Ks
    contact_diff = -(z_contact - 0.82)  # flip sign: lower contact = positive adj

    # BB% — higher walk rate = more pitches per PA = slightly more K opportunities
    bb_diff = bb_pct - 0.085  # league avg ~8.5%

    # Weighted combination → multiplier
    # Each differential is roughly % points, combine them
    raw_adj = (k_diff * 0.35) + (chase_diff * 0.25) + (swstr_diff * 0.20) + (contact_diff * 0.10) + (bb_diff * 0.10)
    multiplier = 1.0 + (raw_adj / LEAGUE_AVG_K_PCT)  # normalize to K% scale

    return max(0.80, min(1.25, multiplier))  # clamp


def project_strikeouts(
    pitcher_stats: dict,
    lineup_stats: dict,
    park_factor: float = PARK_FACTOR_DEFAULT,
    weather: dict | None = None,
    umpire: dict | None = None,
) -> dict:
    """
    Full projection: pitcher xK rate × batters faced × adjustments.
    Returns projected Ks and breakdown.
    """
    weather = weather or {"adjustment": 1.0}
    umpire = umpire or {"adjustment": 1.0}

    # Core projection
    xk_rate = calc_pitcher_xk_rate(pitcher_stats)
    bf = estimate_batters_faced(pitcher_stats)
    lineup_adj = calc_lineup_adjustment(lineup_stats)
    park_adj = park_factor / 100.0  # FanGraphs uses 100 = neutral
    weather_adj = weather.get("adjustment", 1.0)
    ump_adj = umpire.get("adjustment", 1.0)

    pitch_mix_adj = calc_pitch_mix_adjustment(pitcher_stats, lineup_stats)
    raw_ks = xk_rate * bf
    adjusted_ks = raw_ks * lineup_adj * park_adj * weather_adj * ump_adj * pitch_mix_adj

    # Recent form: blend with rolling averages, scaled by sample size
    rolling_3 = pitcher_stats.get("rolling_k_3", 0)
    rolling_5 = pitcher_stats.get("rolling_k_5", 0)
    num_starts = pitcher_stats.get("num_starts", 0)

    if rolling_3 > 0 and rolling_5 > 0 and num_starts >= 3:
        form_ks = rolling_3 * 0.6 + rolling_5 * 0.4
        # Only trust rolling averages once we have 3+ starts.
        # Scale weight gradually: 3 starts=20%, 4=28%, 5=35%, 6+=40%
        rolling_weight = min(0.40, (num_starts - 2) / 4 * 0.40)
        final_ks = adjusted_ks * (1 - rolling_weight) + form_ks * rolling_weight
    else:
        final_ks = adjusted_ks

    # Hard cap — no starter should project above 14 Ks or below 0.5
    final_ks = max(0.5, min(14.0, final_ks))

    return {
        "projected_ks": round(final_ks, 1),
        "pitch_mix_adj": round(pitch_mix_adj, 4),
        "xk_rate": round(xk_rate, 4),
        "batters_faced": bf,
        "lineup_adj": round(lineup_adj, 3),
        "park_adj": round(park_adj, 3),
        "weather_adj": weather_adj,
        "ump_adj": ump_adj,
        "raw_ks": round(raw_ks, 1),
        "adjusted_ks": round(adjusted_ks, 1),
    }


def calc_k_distribution(projected_ks: float, std_dev: float = 2.5) -> dict[int, float]:
    """
    Probability distribution over discrete K totals using a normal approximation.
    Returns {k: probability} for k in 0..9 and 10+ as a bucket.
    P(K = n) = CDF(n + 0.5) - CDF(n - 0.5)
    P(K = 10+) = 1 - CDF(9.5)
    """
    import math

    def cdf(x):
        z = (x - projected_ks) / std_dev
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    dist = {}
    for k in range(0, 10):
        dist[k] = max(0.0, cdf(k + 0.5) - cdf(k - 0.5))
    dist["10+"] = max(0.0, 1.0 - cdf(9.5))

    # Normalize so probabilities sum to 1
    total = sum(dist.values())
    if total > 0:
        dist = {k: round(v / total, 4) for k, v in dist.items()}

    return dist


def calc_hit_rate(projected_ks: float, line: float) -> float:
    """
    Estimate probability of hitting the over based on projected Ks vs line.
    Uses empirical std dev and bias correction derived from logged results.
    Update HIT_RATE_STD_DEV and HIT_RATE_BIAS in config.py as more games are logged.
    """
    if line <= 0:
        return 50.0

    import math
    from config import HIT_RATE_STD_DEV, HIT_RATE_BIAS

    # Apply bias correction: model over-projects by HIT_RATE_BIAS on average
    corrected_proj = projected_ks - HIT_RATE_BIAS
    z = (corrected_proj - line) / HIT_RATE_STD_DEV
    hit_rate = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return round(hit_rate * 100, 1)


def calc_fair_value_odds(hit_rate: float) -> int:
    """
    Convert a hit rate % into American odds (fair value, no vig).
    70% hit rate → -233 (what the line should be at fair value).
    """
    p = max(0.01, min(0.99, hit_rate / 100))
    if p >= 0.5:
        return round(-(p / (1 - p)) * 100)
    else:
        return round(((1 - p) / p) * 100)


def calc_edge(hit_rate: float, odds: int) -> float:
    """
    Calculate edge: model implied probability vs book implied probability.
    Returns edge in percentage points.
    """
    # Convert American odds to implied probability
    if odds > 0:
        book_prob = 100 / (odds + 100)
    elif odds < 0:
        book_prob = abs(odds) / (abs(odds) + 100)
    else:
        book_prob = 0.50

    model_prob = hit_rate / 100
    edge = (model_prob - book_prob) * 100  # in percentage points
    return round(edge, 1)


def score_matchup(
    pitcher_stats: dict,
    lineup_stats: dict,
    odds_data: dict,
    park_factor: float = PARK_FACTOR_DEFAULT,
    weather: dict | None = None,
    umpire: dict | None = None,
) -> dict:
    """
    Score a single matchup. Returns full projection with edge analysis.
    """
    projection = project_strikeouts(
        pitcher_stats, lineup_stats, park_factor, weather, umpire
    )

    line = odds_data.get("line", 0)
    best_over = odds_data.get("best_over")
    best_under = odds_data.get("best_under")

    # Hit rates for over and under
    over_hit_rate = calc_hit_rate(projection["projected_ks"], line)
    under_hit_rate = 100 - over_hit_rate

    # Edge calculation
    over_edge = 0.0
    under_edge = 0.0
    if best_over:
        over_edge = calc_edge(over_hit_rate, best_over["odds"])
    if best_under:
        under_edge = calc_edge(under_hit_rate, best_under["odds"])

    # Determine play direction based on projection vs line
    if line <= 0 or projection["projected_ks"] >= line:
        play = "OVER"
        edge = over_edge
        hit_rate = over_hit_rate
        best_line = best_over
    else:
        play = "UNDER"
        edge = under_edge
        hit_rate = under_hit_rate
        best_line = best_under

    k_dist = calc_k_distribution(projection["projected_ks"])

    return {
        **projection,
        "line": line,
        "play": play,
        "edge": edge,
        "hit_rate": hit_rate,
        "over_hit_rate": over_hit_rate,
        "under_hit_rate": under_hit_rate,
        "over_edge": over_edge,
        "under_edge": under_edge,
        "best_line": best_line,
        "over_odds": odds_data.get("over_odds", []),
        "under_odds": odds_data.get("under_odds", []),
        "k_distribution": k_dist,
    }


if __name__ == "__main__":
    # Test with synthetic data
    pitcher = {
        "swstr_pct": 0.135,
        "csw_pct": 0.32,
        "k_pct": 0.28,
        "xk_pct": 0.27,
        "rolling_k_3": 7.3,
        "rolling_k_5": 6.8,
        "swstr_trend": 0.005,
        "velocity": {"avg_velo": 95.2, "trend": 0.3},
        "days_rest": 5,
        "last_outing_pitches": 92,
    }
    lineup = {
        "team_k_pct": 0.245,
        "o_swing_pct": 0.32,
        "swstr_pct_against": 0.12,
        "z_contact_pct": 0.80,
    }
    odds = {
        "line": 5.5,
        "best_over": {"book": "DraftKings", "odds": -120},
        "best_under": {"book": "FanDuel", "odds": -105},
        "over_odds": [{"book": "DraftKings", "odds": -120}],
        "under_odds": [{"book": "FanDuel", "odds": -105}],
    }

    result = score_matchup(pitcher, lineup, odds, park_factor=102)
    print("Test Matchup Score:")
    for k, v in result.items():
        if k in ("over_odds", "under_odds"):
            continue
        print(f"  {k}: {v}")
