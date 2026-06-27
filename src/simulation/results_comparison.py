"""Compare model predictions against actual World Cup results."""

import pandas as pd

from src.features.engineering import build_features


def _actual_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    elif home_goals < away_goals:
        return "away_win"
    return "draw"


def compare_results(
    actual_results: dict,
    df: pd.DataFrame,
    elo_ratings: dict,
    predict_fn,
    match_date: pd.Timestamp | None = None,
) -> dict:
    """Compare model predictions against actual match results.

    Args:
        actual_results: Dict keyed by group letter, each containing a list of
                       match dicts with home, away, home_goals, away_goals.
        df: Historical match DataFrame for feature engineering.
        elo_ratings: Precomputed Elo ratings dict.
        predict_fn: Model prediction function.
        match_date: Date to use for feature engineering (defaults to 2026-06-11).

    Returns:
        Dict with 'groups', 'knockout', and 'summary' keys.
    """
    if match_date is None:
        match_date = pd.Timestamp("2026-06-11")

    groups = {}
    total_correct = 0
    total_matches = 0

    for group_key, matches in actual_results.items():
        group_matches = []
        group_correct = 0

        for match in matches:
            features = build_features(
                df=df,
                home_team=match["home"],
                away_team=match["away"],
                match_date=match_date,
                neutral=True,
                tournament="FIFA World Cup",
                elo_ratings=elo_ratings,
            )
            X = pd.DataFrame([features])
            probs = predict_fn(X)
            hw = float(probs["home_win"].iloc[0])
            dw = float(probs["draw"].iloc[0])
            aw = float(probs["away_win"].iloc[0])

            prob_map = {"home_win": hw, "draw": dw, "away_win": aw}
            predicted_result = max(prob_map, key=prob_map.get)
            actual = _actual_result(match["home_goals"], match["away_goals"])
            correct = predicted_result == actual

            if correct:
                group_correct += 1

            group_matches.append({
                "home": match["home"],
                "away": match["away"],
                "predicted": {
                    "home_win": round(hw, 4),
                    "draw": round(dw, 4),
                    "away_win": round(aw, 4),
                    "result": predicted_result,
                },
                "actual": {
                    "home_goals": match["home_goals"],
                    "away_goals": match["away_goals"],
                    "result": actual,
                },
                "correct": correct,
            })

        total_correct += group_correct
        total_matches += len(matches)

        groups[group_key] = {
            "matches": group_matches,
            "accuracy": round(group_correct / len(matches) * 100, 1) if matches else 0,
        }

    return {
        "groups": groups,
        "knockout": {},
        "summary": {
            "total_matches": total_matches,
            "correct": total_correct,
            "accuracy": round(total_correct / total_matches * 100, 1) if total_matches else 0,
        },
    }
