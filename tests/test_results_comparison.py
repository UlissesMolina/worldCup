import pandas as pd
import pytest

from src.simulation.results_comparison import compare_results


def _make_predict_fn(home_win=0.5, draw=0.3, away_win=0.2):
    def predict_fn(X):
        return pd.DataFrame([{
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
        }])
    return predict_fn


@pytest.fixture
def fake_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "home_team": ["Brazil", "Germany"],
        "away_team": ["Germany", "Brazil"],
        "home_score": [2, 1],
        "away_score": [1, 1],
        "tournament": ["Friendly", "Friendly"],
        "neutral": [False, False],
    })


@pytest.fixture
def fake_elo(fake_df):
    from src.features.engineering import compute_elo_ratings
    return compute_elo_ratings(fake_df)


def test_compare_results_structure(fake_df, fake_elo):
    actual = {
        "X": [
            {"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1},
            {"home": "Germany", "away": "Brazil", "home_goals": 0, "away_goals": 0},
        ],
    }
    result = compare_results(actual, fake_df, fake_elo, _make_predict_fn())

    assert "groups" in result
    assert "X" in result["groups"]
    group = result["groups"]["X"]
    assert "matches" in group
    assert len(group["matches"]) == 2
    assert "accuracy" in group

    match = group["matches"][0]
    assert "predicted" in match
    assert "actual" in match
    assert "correct" in match
    assert match["predicted"]["home_win"] == 0.5
    assert match["actual"]["home_goals"] == 2
    assert match["actual"]["result"] == "home_win"

    assert "summary" in result
    assert "total_matches" in result["summary"]
    assert "correct" in result["summary"]
    assert "accuracy" in result["summary"]


def test_compare_results_correct_flag(fake_df, fake_elo):
    actual = {
        "X": [
            {"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1},
        ],
    }
    result = compare_results(actual, fake_df, fake_elo, _make_predict_fn())
    assert result["groups"]["X"]["matches"][0]["correct"] is True


def test_compare_results_wrong_flag(fake_df, fake_elo):
    actual = {
        "X": [
            {"home": "Brazil", "away": "Germany", "home_goals": 0, "away_goals": 1},
        ],
    }
    result = compare_results(actual, fake_df, fake_elo, _make_predict_fn())
    assert result["groups"]["X"]["matches"][0]["correct"] is False


def test_compare_results_accuracy(fake_df, fake_elo):
    actual = {
        "X": [
            {"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1},
            {"home": "Germany", "away": "Brazil", "home_goals": 0, "away_goals": 1},
        ],
    }
    result = compare_results(actual, fake_df, fake_elo, _make_predict_fn())
    assert result["summary"]["total_matches"] == 2
    assert result["summary"]["correct"] == 1
    assert result["summary"]["accuracy"] == 50.0
