import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler


CLASS_ORDER = ["away_win", "draw", "home_win"]


class MatchPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)


def train_pytorch(X: pd.DataFrame, y: pd.Series, epochs: int = 100, lr: float = 0.001) -> dict:
    """Train a PyTorch feedforward net. Returns dict with model, scaler, and label encoder."""
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y_encoded = le.transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y_encoded)

    model = MatchPredictor(input_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    return {"model": model, "scaler": scaler, "label_encoder": le}


def predict_pytorch(trained: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Predict probabilities. Returns DataFrame with columns: away_win, draw, home_win."""
    model = trained["model"]
    scaler = trained["scaler"]
    le = trained["label_encoder"]

    X_scaled = scaler.transform(X.values)
    X_tensor = torch.FloatTensor(X_scaled)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()

    return pd.DataFrame(probs, columns=le.classes_)


def save_pytorch(trained: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "model_state": trained["model"].state_dict(),
        "input_dim": trained["model"].net[0].in_features,
        "scaler": trained["scaler"],
        "label_encoder": trained["label_encoder"],
    }
    with open(path, "wb") as f:
        pickle.dump(save_data, f)


def load_pytorch(path: Path) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    model = MatchPredictor(input_dim=data["input_dim"])
    model.load_state_dict(data["model_state"])
    model.eval()
    return {"model": model, "scaler": data["scaler"], "label_encoder": data["label_encoder"]}
