import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder


CLASS_ORDER = ["away_win", "draw", "home_win"]


def evaluate_model(y_true: pd.Series, y_probs: pd.DataFrame) -> dict:
    """Compute log loss and accuracy from true labels and predicted probabilities."""
    le = LabelEncoder()
    le.fit(CLASS_ORDER)

    y_pred_labels = y_probs.idxmax(axis=1)
    acc = accuracy_score(y_true, y_pred_labels)

    ll = log_loss(y_true, y_probs.values, labels=CLASS_ORDER)

    return {"log_loss": ll, "accuracy": acc}


def select_best_model(results: dict[str, dict]) -> str:
    """Select the model with the lowest log loss."""
    return min(results, key=lambda name: results[name]["log_loss"])
