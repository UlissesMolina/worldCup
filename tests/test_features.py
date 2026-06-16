import pandas as pd
import numpy as np
import pytest
from src.features.engineering import (
    compute_team_form, compute_h2h, compute_rest_days, compute_goal_diff_form,
    encode_tournament, build_features, build_dataset,
    compute_elo_ratings, get_elo_rating, compute_goals_scored_rate, compute_goals_conceded_rate,
)


def test_compute_team_form_win_rate(sample_matches):
    df = sample_matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    form = compute_team_form(df, "Brazil", pd.Timestamp("2023-06-10"), n=5)
    assert "win_rate" in form
    assert 0.0 <= form["win_rate"] <= 1.0


def test_compute_team_form_no_history():
    df = pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score"])
    form = compute_team_form(df, "Brazil", pd.Timestamp("2020-01-01"), n=5)
    assert form["win_rate"] == 0.0


def test_compute_team_form_respects_date_cutoff(sample_matches):
    form = compute_team_form(sample_matches, "Brazil", pd.Timestamp("2020-06-20"), n=10)
    assert form["matches_played"] == 2


def test_compute_h2h_returns_win_rate(sample_matches):
    h2h = compute_h2h(sample_matches, "Brazil", "Argentina", pd.Timestamp("2023-06-10"), n=10)
    assert "h2h_win_rate" in h2h
    assert 0.0 <= h2h["h2h_win_rate"] <= 1.0


def test_compute_h2h_no_meetings():
    df = pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score"])
    h2h = compute_h2h(df, "Brazil", "Japan", pd.Timestamp("2023-01-01"), n=10)
    assert h2h["h2h_win_rate"] == 0.0
    assert h2h["h2h_matches"] == 0


def test_compute_rest_days(sample_matches):
    days = compute_rest_days(sample_matches, "Brazil", pd.Timestamp("2020-06-20"))
    assert days > 90


def test_compute_rest_days_no_history():
    df = pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    days = compute_rest_days(df, "Brazil", pd.Timestamp("2020-01-01"))
    assert days == -1


def test_compute_goal_diff_form(sample_matches):
    gd = compute_goal_diff_form(sample_matches, "Brazil", pd.Timestamp("2023-06-10"), n=5)
    assert isinstance(gd, float)


def test_encode_tournament():
    assert encode_tournament("FIFA World Cup") == "major"
    assert encode_tournament("Friendly") == "friendly"
    assert encode_tournament("FIFA World Cup qualification") == "qualifier"
    assert encode_tournament("Copa América") == "major"
    assert encode_tournament("Some Unknown Tournament") == "other"


def test_compute_elo_ratings(sample_matches):
    elo = compute_elo_ratings(sample_matches)
    # All teams should have entries
    assert any(t == "Brazil" for t, _ in elo.keys())
    assert any(t == "Germany" for t, _ in elo.keys())
    # Latest ratings should differ from 1500 after matches
    assert elo[("Brazil", None)] != 1500.0
    assert elo[("Germany", None)] != 1500.0


def test_get_elo_rating(sample_matches):
    elo = compute_elo_ratings(sample_matches)
    # Before any match, should get default 1500
    rating = get_elo_rating(elo, "Brazil", pd.Timestamp("2019-01-01"))
    assert rating == 1500.0
    # After some matches, should differ
    rating = get_elo_rating(elo, "Brazil", pd.Timestamp("2023-06-10"))
    assert rating != 1500.0


def test_compute_goals_scored_rate(sample_matches):
    rate = compute_goals_scored_rate(sample_matches, "Brazil", pd.Timestamp("2023-06-10"), n=5)
    assert isinstance(rate, float)
    assert rate >= 0.0


def test_compute_goals_scored_rate_no_history():
    df = pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    rate = compute_goals_scored_rate(df, "Brazil", pd.Timestamp("2020-01-01"), n=5)
    assert rate == 0.0


def test_compute_goals_conceded_rate(sample_matches):
    rate = compute_goals_conceded_rate(sample_matches, "Brazil", pd.Timestamp("2023-06-10"), n=5)
    assert isinstance(rate, float)
    assert rate >= 0.0


def test_compute_goals_conceded_rate_no_history():
    df = pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    rate = compute_goals_conceded_rate(df, "Brazil", pd.Timestamp("2020-01-01"), n=5)
    assert rate == 0.0


def test_build_features_returns_expected_keys(sample_matches):
    features = build_features(sample_matches, "Brazil", "Germany", pd.Timestamp("2023-06-10"), neutral=True)
    expected_keys = [
        "home_form_5", "home_form_10", "away_form_5", "away_form_10",
        "h2h_win_rate", "home_rest_days", "away_rest_days",
        "home_goal_diff", "away_goal_diff", "neutral",
        "tournament_friendly", "tournament_qualifier", "tournament_major", "tournament_other",
        "home_elo", "away_elo", "elo_diff",
        "home_goals_scored_rate", "home_goals_conceded_rate",
        "away_goals_scored_rate", "away_goals_conceded_rate",
    ]
    for key in expected_keys:
        assert key in features, f"Missing feature: {key}"
    assert len(features) == 21


def test_build_features_values_are_numeric(sample_matches):
    features = build_features(sample_matches, "Brazil", "Germany", pd.Timestamp("2023-06-10"), neutral=False)
    for key, val in features.items():
        assert isinstance(val, (int, float, np.integer, np.floating)), f"{key} is {type(val)}"


def test_build_dataset_returns_X_and_y(sample_matches):
    from src.data.process import add_target
    from src.features.engineering import build_dataset
    df = add_target(sample_matches)
    X, y = build_dataset(df)
    assert len(X) == len(y)
    assert len(X) > 0
    assert set(y.unique()).issubset({"home_win", "draw", "away_win"})
    assert "home_form_5" in X.columns
    assert "neutral" in X.columns
