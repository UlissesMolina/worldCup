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


def compute_elo_ratings(df: pd.DataFrame, k: int = 20) -> dict:
    """Compute Elo ratings for all teams from chronological match history.

    Returns a dict mapping (team, date) → elo rating *before* that match.
    Also stores a running 'latest' keyed as (team, None) for lookup convenience.
    """
    elo = {}  # team -> current elo
    history = {}  # (team, date) -> elo before match

    for _, row in df.sort_values("date").iterrows():
        home, away = row["home_team"], row["away_team"]
        date = row["date"]

        home_elo = elo.get(home, 1500.0)
        away_elo = elo.get(away, 1500.0)

        # Record pre-match ratings
        history[(home, date)] = home_elo
        history[(away, date)] = away_elo

        # Expected scores
        exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400))
        exp_away = 1.0 - exp_home

        # Actual scores
        if row["home_score"] > row["away_score"]:
            actual_home, actual_away = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        elo[home] = home_elo + k * (actual_home - exp_home)
        elo[away] = away_elo + k * (actual_away - exp_away)

    # Store latest ratings
    for team, rating in elo.items():
        history[(team, None)] = rating

    return history


def get_elo_rating(elo_dict: dict, team: str, before_date: pd.Timestamp) -> float:
    """Look up team's Elo rating just before a given date.

    Finds the most recent entry for the team before before_date.
    Returns 1500.0 if no history exists.
    """
    best_date = None
    best_elo = 1500.0

    for (t, d), rating in elo_dict.items():
        if t != team or d is None:
            continue
        if d < before_date:
            if best_date is None or d > best_date:
                best_date = d
                best_elo = rating

    return best_elo


def compute_goals_scored_rate(df: pd.DataFrame, team: str, before_date: pd.Timestamp, n: int = 5) -> float:
    """Average goals scored per match over last n games before a given date."""
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < before_date)
    )
    team_matches = df[mask].sort_values("date").tail(n)

    if len(team_matches) == 0:
        return 0.0

    total = 0
    for _, row in team_matches.iterrows():
        if row["home_team"] == team:
            total += row["home_score"]
        else:
            total += row["away_score"]

    return total / len(team_matches)


def compute_goals_conceded_rate(df: pd.DataFrame, team: str, before_date: pd.Timestamp, n: int = 5) -> float:
    """Average goals conceded per match over last n games before a given date."""
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < before_date)
    )
    team_matches = df[mask].sort_values("date").tail(n)

    if len(team_matches) == 0:
        return 0.0

    total = 0
    for _, row in team_matches.iterrows():
        if row["home_team"] == team:
            total += row["away_score"]
        else:
            total += row["home_score"]

    return total / len(team_matches)


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
    elo_ratings: dict | None = None,
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

    # Elo ratings
    if elo_ratings is not None:
        home_elo = get_elo_rating(elo_ratings, home_team, match_date)
        away_elo = get_elo_rating(elo_ratings, away_team, match_date)
    else:
        home_elo = 1500.0
        away_elo = 1500.0

    # Goal scoring/conceding rates
    home_goals_scored = compute_goals_scored_rate(df, home_team, match_date, n=5)
    home_goals_conceded = compute_goals_conceded_rate(df, home_team, match_date, n=5)
    away_goals_scored = compute_goals_scored_rate(df, away_team, match_date, n=5)
    away_goals_conceded = compute_goals_conceded_rate(df, away_team, match_date, n=5)

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
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_goals_scored_rate": home_goals_scored,
        "home_goals_conceded_rate": home_goals_conceded,
        "away_goals_scored_rate": away_goals_scored,
        "away_goals_conceded_rate": away_goals_conceded,
    }


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target vector y from a processed match DataFrame.

    df must have a 'target' column (from add_target) and be sorted by date.
    """
    elo_ratings = compute_elo_ratings(df)

    rows = []
    for idx, match in df.iterrows():
        features = build_features(
            df=df,
            home_team=match["home_team"],
            away_team=match["away_team"],
            match_date=match["date"],
            neutral=bool(match["neutral"]),
            tournament=match.get("tournament", "other"),
            elo_ratings=elo_ratings,
        )
        rows.append(features)

    X = pd.DataFrame(rows)
    y = df["target"].reset_index(drop=True)
    return X, y
