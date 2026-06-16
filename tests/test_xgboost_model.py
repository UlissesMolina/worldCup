import numpy as np
import pandas as pd
import pytest
from src.models.xgboost_model import train_xgboost, predict_xgboost


@pytest.fixture
def dummy_training_data():
    """Tiny dataset for fast model tests."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "home_form_5": np.random.rand(n),
        "home_form_10": np.random.rand(n),
        "away_form_5": np.random.rand(n),
        "away_form_10": np.random.rand(n),
        "h2h_win_rate": np.random.rand(n),
        "home_rest_days": np.random.randint(1, 60, n),
        "away_rest_days": np.random.randint(1, 60, n),
        "home_goal_diff": np.random.randn(n),
        "away_goal_diff": np.random.randn(n),
        "neutral": np.random.randint(0, 2, n),
        "tournament_friendly": np.random.randint(0, 2, n),
        "tournament_qualifier": np.random.randint(0, 2, n),
        "tournament_major": np.random.randint(0, 2, n),
        "tournament_other": np.random.randint(0, 2, n),
    })
    y = pd.Series(np.random.choice(["home_win", "draw", "away_win"], n))
    return X, y


def test_train_xgboost_returns_model(dummy_training_data):
    X, y = dummy_training_data
    model = train_xgboost(X, y)
    assert model is not None


def test_predict_xgboost_returns_probabilities(dummy_training_data):
    X, y = dummy_training_data
    model = train_xgboost(X, y)
    probs = predict_xgboost(model, X.iloc[[0]])
    assert probs.shape == (1, 3)
    assert abs(probs.sum().sum() - 1.0) < 1e-6


def test_predict_xgboost_class_order(dummy_training_data):
    X, y = dummy_training_data
    model = train_xgboost(X, y)
    probs = predict_xgboost(model, X.iloc[[0]])
    assert list(probs.columns) == ["away_win", "draw", "home_win"]
