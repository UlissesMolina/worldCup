"""Tournament simulation: group stage, standings, seeding, knockout."""

import pandas as pd

from src.features.engineering import build_features
from src.data.groups import FIXED_R32, WINNER_VS_THIRD_POOLS


def estimate_scoreline(
    result: str, home_win_prob: float, draw_prob: float, away_win_prob: float
) -> tuple[int, int]:
    """Estimate a scoreline from the predicted result and probabilities.

    Returns (home_goals, away_goals). Uses probability magnitude to scale margin.
    """
    if result == "home_win":
        margin = max(1, round(home_win_prob * 3))
        home = 1 + margin
        away = max(0, home - margin)
        return (home, away)
    elif result == "away_win":
        margin = max(1, round(away_win_prob * 3))
        away = 1 + margin
        home = max(0, away - margin)
        return (home, away)
    else:  # draw
        goals = max(0, round(draw_prob * 2))
        return (goals, goals)


def predict_match(
    home_team: str,
    away_team: str,
    df: pd.DataFrame,
    elo_ratings: dict,
    predict_fn,
    match_date: pd.Timestamp,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
) -> dict:
    """Predict a single match. Returns dict with teams, probs, result, goals."""
    features = build_features(
        df=df,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        neutral=neutral,
        tournament=tournament,
        elo_ratings=elo_ratings,
    )
    X = pd.DataFrame([features])
    probs = predict_fn(X)
    home_win_prob = float(probs["home_win"].iloc[0])
    draw_prob = float(probs["draw"].iloc[0])
    away_win_prob = float(probs["away_win"].iloc[0])

    # Determine result: highest probability wins
    prob_map = {"home_win": home_win_prob, "draw": draw_prob, "away_win": away_win_prob}
    result = max(prob_map, key=prob_map.get)

    home_goals, away_goals = estimate_scoreline(result, home_win_prob, draw_prob, away_win_prob)

    return {
        "home": home_team,
        "away": away_team,
        "home_win": round(home_win_prob, 4),
        "draw": round(draw_prob, 4),
        "away_win": round(away_win_prob, 4),
        "result": result,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def compute_group_standings(teams: list[str], matches: list[dict]) -> list[dict]:
    """Compute group standings from match results.

    Ranking: points -> goal difference -> goals scored -> alphabetical.
    Returns list of dicts sorted by rank, each with pos, team, w, d, l, gf, ga, gd, pts.
    """
    stats = {t: {"w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0} for t in teams}

    for m in matches:
        h, a = m["home"], m["away"]
        hg, ag = m["home_goals"], m["away_goals"]
        stats[h]["gf"] += hg
        stats[h]["ga"] += ag
        stats[a]["gf"] += ag
        stats[a]["ga"] += hg

        if m["result"] == "home_win":
            stats[h]["w"] += 1
            stats[a]["l"] += 1
        elif m["result"] == "away_win":
            stats[a]["w"] += 1
            stats[h]["l"] += 1
        else:
            stats[h]["d"] += 1
            stats[a]["d"] += 1

    rows = []
    for team, s in stats.items():
        pts = s["w"] * 3 + s["d"]
        gd = s["gf"] - s["ga"]
        rows.append({
            "team": team,
            "w": s["w"], "d": s["d"], "l": s["l"],
            "gf": s["gf"], "ga": s["ga"], "gd": gd, "pts": pts,
        })

    # Sort: pts desc, gd desc, gf desc, team asc (alphabetical tiebreak)
    rows.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], r["team"]))

    for i, row in enumerate(rows):
        row["pos"] = i + 1

    return rows


def simulate_group_stage(
    groups: dict[str, list[str]],
    df: pd.DataFrame,
    elo_ratings: dict,
    predict_fn,
    match_date: pd.Timestamp,
) -> dict:
    """Simulate all group stage matches.

    Returns dict keyed by group letter, each with 'standings' and 'matches'.
    """
    from itertools import combinations

    result = {}
    for group_letter, teams in groups.items():
        matches = []
        # Round-robin: each pair plays once. First-listed team is "home".
        for home, away in combinations(teams, 2):
            match = predict_match(
                home_team=home,
                away_team=away,
                df=df,
                elo_ratings=elo_ratings,
                predict_fn=predict_fn,
                match_date=match_date,
                neutral=True,
                tournament="FIFA World Cup",
            )
            matches.append(match)

        standings = compute_group_standings(teams, matches)
        # Mark who advances: pos 1 and 2 always advance
        for row in standings:
            row["advanced"] = row["pos"] <= 2

        result[group_letter] = {
            "standings": standings,
            "matches": matches,
        }

    return result


def rank_third_place(group_results: dict) -> list[dict]:
    """Rank all 12 third-place teams. Top 8 advance."""
    thirds = []
    for group_letter, data in group_results.items():
        for row in data["standings"]:
            if row["pos"] == 3:
                entry = {**row, "group": group_letter}
                thirds.append(entry)
                break

    thirds.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], r["team"]))

    for i, entry in enumerate(thirds):
        entry["advanced"] = i < 8

    return thirds


def _get_team_by_position(group_results: dict, group: str, pos: int) -> str:
    """Get team name by group letter and position (1, 2, 3)."""
    for row in group_results[group]["standings"]:
        if row["pos"] == pos:
            return row["team"]
    raise ValueError(f"No team at position {pos} in group {group}")


def seed_bracket(
    group_results: dict, third_place_ranked: list[dict]
) -> list[dict]:
    """Create 16 R32 matchups from group results and third-place ranking.

    Returns list of 16 dicts, each with 'home' and 'away' team names.
    Order: indices 0-7 left bracket, 8-15 right bracket.
    """
    fixed_matches = []
    for home_code, away_code in FIXED_R32:
        home_pos = int(home_code[0])
        home_group = home_code[1]
        away_pos = int(away_code[0])
        away_group = away_code[1]
        home_team = _get_team_by_position(group_results, home_group, home_pos)
        away_team = _get_team_by_position(group_results, away_group, away_pos)
        fixed_matches.append({"home": home_team, "away": away_team})

    advancing_thirds = {
        entry["group"]: entry["team"]
        for entry in third_place_ranked
        if entry["advanced"]
    }
    assigned_groups = set()

    winner_matches = []
    for winner_group in sorted(WINNER_VS_THIRD_POOLS.keys()):
        pool = WINNER_VS_THIRD_POOLS[winner_group]
        winner_team = _get_team_by_position(group_results, winner_group, 1)
        third_team = None
        for candidate_group in pool:
            if candidate_group in advancing_thirds and candidate_group not in assigned_groups:
                third_team = advancing_thirds[candidate_group]
                assigned_groups.add(candidate_group)
                break
        if third_team is None:
            for g, t in advancing_thirds.items():
                if g not in assigned_groups:
                    third_team = t
                    assigned_groups.add(g)
                    break
        winner_matches.append({"home": winner_team, "away": third_team})

    # Interleave: Left bracket fixed[0],winner[0],...; Right bracket winner[4],fixed[4],...
    r32 = []
    for i in range(4):
        r32.append(fixed_matches[i])
        r32.append(winner_matches[i])
    for i in range(4):
        r32.append(winner_matches[4 + i])
        r32.append(fixed_matches[4 + i])

    return r32


def simulate_knockout_round(
    matchups: list[dict],
    df: pd.DataFrame,
    elo_ratings: dict,
    predict_fn,
    match_date: pd.Timestamp,
) -> list[dict]:
    """Simulate a knockout round. No draws — higher win prob advances."""
    results = []
    for matchup in matchups:
        match = predict_match(
            home_team=matchup["home"],
            away_team=matchup["away"],
            df=df,
            elo_ratings=elo_ratings,
            predict_fn=predict_fn,
            match_date=match_date,
            neutral=True,
            tournament="FIFA World Cup",
        )
        if match["result"] == "home_win":
            match["winner"] = match["home"]
        elif match["result"] == "away_win":
            match["winner"] = match["away"]
        else:
            if match["home_win"] >= match["away_win"]:
                match["winner"] = match["home"]
            else:
                match["winner"] = match["away"]
        results.append(match)
    return results


def simulate_tournament(
    df: pd.DataFrame,
    elo_ratings: dict,
    predict_fn,
    match_date: pd.Timestamp | None = None,
) -> dict:
    """Simulate the full 2026 World Cup tournament."""
    from src.data.groups import GROUPS

    if match_date is None:
        match_date = pd.Timestamp("2026-06-11")

    # 1. Group stage
    group_results = simulate_group_stage(
        GROUPS, df, elo_ratings, predict_fn, match_date
    )

    # 2. Third-place ranking
    third_place = rank_third_place(group_results)

    # Update group standings with third-place advancement
    for entry in third_place:
        if entry["advanced"]:
            group = entry["group"]
            for row in group_results[group]["standings"]:
                if row["pos"] == 3:
                    row["advanced"] = True

    # 3. Seed bracket
    r32_matchups = seed_bracket(group_results, third_place)

    # 4. Knockout rounds
    r32 = simulate_knockout_round(r32_matchups, df, elo_ratings, predict_fn, match_date)

    r16_matchups = [
        {"home": r32[i]["winner"], "away": r32[i + 1]["winner"]}
        for i in range(0, 16, 2)
    ]
    r16 = simulate_knockout_round(r16_matchups, df, elo_ratings, predict_fn, match_date)

    qf_matchups = [
        {"home": r16[i]["winner"], "away": r16[i + 1]["winner"]}
        for i in range(0, 8, 2)
    ]
    qf = simulate_knockout_round(qf_matchups, df, elo_ratings, predict_fn, match_date)

    sf_matchups = [
        {"home": qf[i]["winner"], "away": qf[i + 1]["winner"]}
        for i in range(0, 4, 2)
    ]
    sf = simulate_knockout_round(sf_matchups, df, elo_ratings, predict_fn, match_date)

    final_matchup = [{"home": sf[0]["winner"], "away": sf[1]["winner"]}]
    final = simulate_knockout_round(final_matchup, df, elo_ratings, predict_fn, match_date)

    return {
        "groups": group_results,
        "knockout": {
            "r32": r32,
            "r16": r16,
            "qf": qf,
            "sf": sf,
            "final": final[0],
        },
        "champion": final[0]["winner"],
    }
