import numpy as np
import pandas as pd
import pytest
from src.models.evaluate import evaluate_model, select_best_model


@pytest.fixture
def dummy_predictions():
    y_true = pd.Series(["home_win", "draw", "away_win", "home_win", "home_win"])
    y_probs = pd.DataFrame({
        "away_win": [0.1, 0.2, 0.7, 0.1, 0.2],
        "draw": [0.2, 0.5, 0.2, 0.1, 0.1],
        "home_win": [0.7, 0.3, 0.1, 0.8, 0.7],
    })
    return y_true, y_probs


def test_evaluate_model_returns_metrics(dummy_predictions):
    y_true, y_probs = dummy_predictions
    metrics = evaluate_model(y_true, y_probs)
    assert "log_loss" in metrics
    assert "accuracy" in metrics
    assert metrics["log_loss"] > 0
    assert 0 <= metrics["accuracy"] <= 1


def test_select_best_model():
    results = {
        "xgboost": {"log_loss": 0.85, "accuracy": 0.55},
        "pytorch": {"log_loss": 0.90, "accuracy": 0.52},
    }
    best = select_best_model(results)
    assert best == "xgboost"


def test_select_best_model_pytorch_wins():
    results = {
        "xgboost": {"log_loss": 0.95, "accuracy": 0.50},
        "pytorch": {"log_loss": 0.80, "accuracy": 0.58},
    }
    best = select_best_model(results)
    assert best == "pytorch"


def test_evaluate_model_returns_per_class_metrics(dummy_predictions):
    y_true, y_probs = dummy_predictions
    metrics = evaluate_model(y_true, y_probs)

    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics

    report = metrics["classification_report"]
    for cls in ["away_win", "draw", "home_win"]:
        assert cls in report
        assert "precision" in report[cls]
        assert "recall" in report[cls]
        assert "f1-score" in report[cls]

    cm = metrics["confusion_matrix"]
    assert len(cm) == 3
    assert len(cm[0]) == 3
