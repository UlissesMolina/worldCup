import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


CLASS_ORDER = ["away_win", "draw", "home_win"]


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train an XGBoost classifier. Returns a dict with model and label encoder."""
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y_encoded = le.transform(y)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X, y_encoded)

    return {"model": model, "label_encoder": le}


def predict_xgboost(trained: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Predict probabilities. Returns DataFrame with columns: away_win, draw, home_win."""
    model = trained["model"]
    le = trained["label_encoder"]
    probs = model.predict_proba(X)
    return pd.DataFrame(probs, columns=le.classes_)


def save_xgboost(trained: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(trained, f)


def load_xgboost(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
