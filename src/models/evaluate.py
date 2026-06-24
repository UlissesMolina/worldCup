import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


CLASS_ORDER = ["away_win", "draw", "home_win"]


def evaluate_model(y_true: pd.Series, y_probs: pd.DataFrame) -> dict:
    """Compute log loss, accuracy, per-class report, and confusion matrix."""
    le = LabelEncoder()
    le.fit(CLASS_ORDER)

    y_pred_labels = y_probs.idxmax(axis=1)
    acc = accuracy_score(y_true, y_pred_labels)

    ll = log_loss(y_true, y_probs.values, labels=CLASS_ORDER)

    report = classification_report(
        y_true, y_pred_labels, target_names=CLASS_ORDER,
        labels=CLASS_ORDER, output_dict=True, zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred_labels, labels=CLASS_ORDER).tolist()

    return {
        "log_loss": ll,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def select_best_model(results: dict[str, dict]) -> str:
    """Select the model with the lowest log loss."""
    return min(results, key=lambda name: results[name]["log_loss"])
