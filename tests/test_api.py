from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features.engineering import compute_elo_ratings
from src.data.groups import GROUPS


@pytest.fixture
def mock_model_and_data():
    """Patch model loading and data so we can test the endpoint without real artifacts."""
    fake_df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "home_team": ["Brazil", "Germany"],
        "away_team": ["Germany", "Brazil"],
        "home_score": [2, 1],
        "away_score": [1, 1],
        "tournament": ["Friendly", "Friendly"],
        "neutral": [False, False],
    })

    fake_probs = pd.DataFrame({
        "away_win": [0.25],
        "draw": [0.30],
        "home_win": [0.45],
    })

    with patch("src.api.main.load_model_and_data") as mock_load:
        mock_load.return_value = {
            "predict_fn": lambda X: fake_probs,
            "df": fake_df,
            "model_type": "xgboost",
            "teams": {"Brazil", "Germany", "Argentina"},
            "elo_ratings": compute_elo_ratings(fake_df),
        }
        from src.api.main import app
        with TestClient(app) as client:
            yield client


def test_predict_returns_probabilities(mock_model_and_data):
    client = mock_model_and_data
    resp = client.post("/predict", json={
        "home_team": "Brazil",
        "away_team": "Germany",
        "neutral_venue": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "home_win" in data
    assert "draw" in data
    assert "away_win" in data
    assert "model_used" in data
    assert abs(data["home_win"] + data["draw"] + data["away_win"] - 1.0) < 1e-4


def test_predict_unknown_team(mock_model_and_data):
    client = mock_model_and_data
    resp = client.post("/predict", json={
        "home_team": "Atlantis",
        "away_team": "Germany",
        "neutral_venue": False,
    })
    assert resp.status_code == 422


def test_predict_same_team(mock_model_and_data):
    client = mock_model_and_data
    resp = client.post("/predict", json={
        "home_team": "Brazil",
        "away_team": "Brazil",
        "neutral_venue": False,
    })
    assert resp.status_code == 422


@pytest.fixture
def mock_bracket_client():
    """Client with all 48 World Cup teams for bracket simulation."""
    all_teams = set()
    for teams in GROUPS.values():
        all_teams.update(teams)

    rows = []
    teams_list = sorted(all_teams)
    for i in range(len(teams_list)):
        for j in range(i + 1, min(i + 3, len(teams_list))):
            rows.append({
                "date": pd.to_datetime("2020-01-01") + pd.Timedelta(days=i + j),
                "home_team": teams_list[i],
                "away_team": teams_list[j],
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "neutral": False,
            })
    fake_df = pd.DataFrame(rows)

    fake_probs = pd.DataFrame({
        "away_win": [0.25],
        "draw": [0.30],
        "home_win": [0.45],
    })

    with patch("src.api.main.load_model_and_data") as mock_load:
        mock_load.return_value = {
            "predict_fn": lambda X: fake_probs,
            "df": fake_df,
            "model_type": "xgboost",
            "teams": all_teams,
            "elo_ratings": compute_elo_ratings(fake_df),
        }
        from src.api.main import app
        with TestClient(app) as client:
            yield client


def test_simulate_bracket_returns_monte_carlo(mock_bracket_client):
    client = mock_bracket_client
    resp = client.post("/simulate-bracket?n_simulations=50")
    assert resp.status_code == 200
    data = resp.json()

    assert "teams" in data
    assert "simulations" in data
    assert data["simulations"] == 50

    # Should have all 48 teams
    assert len(data["teams"]) == 48

    stages = ["group_pct", "r32_pct", "r16_pct", "qf_pct", "sf_pct", "final_pct", "champion_pct"]
    for team, team_data in data["teams"].items():
        for stage in stages:
            assert stage in team_data
            assert 0 <= team_data[stage] <= 100
