import numpy as np
import pandas as pd
import pytest
from src.models.pytorch_model import train_pytorch, predict_pytorch


@pytest.fixture
def dummy_training_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "home_form_5": np.random.rand(n),
        "home_form_10": np.random.rand(n),
        "away_form_5": np.random.rand(n),
        "away_form_10": np.random.rand(n),
        "h2h_win_rate": np.random.rand(n),
        "home_rest_days": np.random.randint(1, 60, n).astype(float),
        "away_rest_days": np.random.randint(1, 60, n).astype(float),
        "home_goal_diff": np.random.randn(n),
        "away_goal_diff": np.random.randn(n),
        "neutral": np.random.randint(0, 2, n).astype(float),
        "tournament_friendly": np.random.randint(0, 2, n).astype(float),
        "tournament_qualifier": np.random.randint(0, 2, n).astype(float),
        "tournament_major": np.random.randint(0, 2, n).astype(float),
        "tournament_other": np.random.randint(0, 2, n).astype(float),
    })
    y = pd.Series(np.random.choice(["home_win", "draw", "away_win"], n))
    return X, y


def test_train_pytorch_returns_model(dummy_training_data):
    X, y = dummy_training_data
    trained = train_pytorch(X, y, epochs=5)
    assert trained is not None
    assert "model" in trained
    assert "scaler" in trained


def test_predict_pytorch_returns_probabilities(dummy_training_data):
    X, y = dummy_training_data
    trained = train_pytorch(X, y, epochs=5)
    probs = predict_pytorch(trained, X.iloc[[0]])
    assert probs.shape == (1, 3)
    assert abs(probs.values.sum() - 1.0) < 1e-4


def test_predict_pytorch_class_order(dummy_training_data):
    X, y = dummy_training_data
    trained = train_pytorch(X, y, epochs=5)
    probs = predict_pytorch(trained, X.iloc[[0]])
    assert list(probs.columns) == ["away_win", "draw", "home_win"]
