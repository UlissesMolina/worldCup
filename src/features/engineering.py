import pandas as pd
import numpy as np


def compute_team_form(df: pd.DataFrame, team: str, before_date: pd.Timestamp, n: int = 5) -> dict:
    """Compute win rate for a team over their last n matches before a given date."""
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < before_date)
    )
    team_matches = df[mask].sort_values("date").tail(n)

    if len(team_matches) == 0:
        return {"win_rate": 0.0, "matches_played": 0}

    wins = 0
    for _, row in team_matches.iterrows():
        if row["home_team"] == team and row["home_score"] > row["away_score"]:
            wins += 1
        elif row["away_team"] == team and row["away_score"] > row["home_score"]:
            wins += 1

    return {
        "win_rate": wins / len(team_matches),
        "matches_played": len(team_matches),
    }


def compute_h2h(df: pd.DataFrame, team_a: str, team_b: str, before_date: pd.Timestamp, n: int = 10) -> dict:
    """Compute head-to-head win rate for team_a against team_b."""
    mask = (
        (
            ((df["home_team"] == team_a) & (df["away_team"] == team_b))
            | ((df["home_team"] == team_b) & (df["away_team"] == team_a))
        )
        & (df["date"] < before_date)
    )
    meetings = df[mask].sort_values("date").tail(n)

    if len(meetings) == 0:
        return {"h2h_win_rate": 0.0, "h2h_matches": 0}

    wins = 0
    for _, row in meetings.iterrows():
        if row["home_team"] == team_a and row["home_score"] > row["away_score"]:
            wins += 1
        elif row["away_team"] == team_a and row["away_score"] > row["home_score"]:
            wins += 1

    return {
        "h2h_win_rate": wins / len(meetings),
        "h2h_matches": len(meetings),
    }


def compute_rest_days(df: pd.DataFrame, team: str, before_date: pd.Timestamp) -> int:
    """Days since team's last match before the given date. Returns -1 if no history."""
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < before_date)
    )
    team_matches = df[mask].sort_values("date")

    if len(team_matches) == 0:
        return -1

    last_match_date = team_matches.iloc[-1]["date"]
    return (before_date - last_match_date).days


def compute_goal_diff_form(df: pd.DataFrame, team: str, before_date: pd.Timestamp, n: int = 5) -> float:
    """Average goal difference over last n matches for a team. Positive = scoring more than conceding."""
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < before_date)
    )
    team_matches = df[mask].sort_values("date").tail(n)

    if len(team_matches) == 0:
        return 0.0

    total_diff = 0
    for _, row in team_matches.iterrows():
        if row["home_team"] == team:
            total_diff += row["home_score"] - row["away_score"]
        else:
            total_diff += row["away_score"] - row["home_score"]

    return total_diff / len(team_matches)


MAJOR_TOURNAMENTS = {
    "FIFA World Cup", "Copa América", "UEFA Euro", "AFC Asian Cup",
    "African Cup of Nations", "Gold Cup", "Confederations Cup",
    "UEFA Nations League",
}

QUALIFIER_KEYWORDS = {"qualification", "qualifiers"}


def encode_tournament(tournament: str) -> str:
    """Map tournament name to category: friendly, qualifier, major, other."""
    lower = tournament.lower()
    if lower == "friendly":
        return "friendly"
    if any(kw in lower for kw in QUALIFIER_KEYWORDS):
        return "qualifier"
    if tournament in MAJOR_TOURNAMENTS:
        return "major"
    return "other"


def build_features(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    neutral: bool,
    tournament: str = "other",
) -> dict:
    """Build the full feature dict for a single match prediction."""
    home_form_5 = compute_team_form(df, home_team, match_date, n=5)
    home_form_10 = compute_team_form(df, home_team, match_date, n=10)
    away_form_5 = compute_team_form(df, away_team, match_date, n=5)
    away_form_10 = compute_team_form(df, away_team, match_date, n=10)

    h2h = compute_h2h(df, home_team, away_team, match_date)

    home_rest = compute_rest_days(df, home_team, match_date)
    away_rest = compute_rest_days(df, away_team, match_date)

    home_gd = compute_goal_diff_form(df, home_team, match_date, n=5)
    away_gd = compute_goal_diff_form(df, away_team, match_date, n=5)

    tourn_cat = encode_tournament(tournament)

    return {
        "home_form_5": home_form_5["win_rate"],
        "home_form_10": home_form_10["win_rate"],
        "away_form_5": away_form_5["win_rate"],
        "away_form_10": away_form_10["win_rate"],
        "h2h_win_rate": h2h["h2h_win_rate"],
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "home_goal_diff": home_gd,
        "away_goal_diff": away_gd,
        "neutral": int(neutral),
        "tournament_friendly": int(tourn_cat == "friendly"),
        "tournament_qualifier": int(tourn_cat == "qualifier"),
        "tournament_major": int(tourn_cat == "major"),
        "tournament_other": int(tourn_cat == "other"),
    }


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target vector y from a processed match DataFrame.

    df must have a 'target' column (from add_target) and be sorted by date.
    """
    rows = []
    for idx, match in df.iterrows():
        features = build_features(
            df=df,
            home_team=match["home_team"],
            away_team=match["away_team"],
            match_date=match["date"],
            neutral=bool(match["neutral"]),
            tournament=match.get("tournament", "other"),
        )
        rows.append(features)

    X = pd.DataFrame(rows)
    y = df["target"].reset_index(drop=True)
    return X, y
