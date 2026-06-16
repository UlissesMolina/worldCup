from pathlib import Path
from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictionRequest, PredictionResponse
from src.features.engineering import build_features
from src.models.xgboost_model import load_xgboost, predict_xgboost
from src.models.pytorch_model import load_pytorch, predict_pytorch


ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

_state = {}


def load_model_and_data() -> dict:
    """Load model artifact and historical data. Called once on startup."""
    meta_path = MODELS_DIR / "metadata.txt"
    meta = dict(line.split("=") for line in meta_path.read_text().strip().splitlines())
    model_type = meta["model_type"]

    model_path = MODELS_DIR / "model.pkl"
    if model_type == "xgboost":
        trained = load_xgboost(model_path)
        predict_fn = lambda X: predict_xgboost(trained, X)
    else:
        trained = load_pytorch(model_path)
        predict_fn = lambda X: predict_pytorch(trained, X)

    df = pd.read_parquet(DATA_DIR / "matches.parquet")
    teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

    return {
        "predict_fn": predict_fn,
        "df": df,
        "model_type": model_type,
        "teams": teams,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.update(load_model_and_data())
    yield
    _state.clear()


app = FastAPI(title="World Cup Predictor", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if req.home_team not in _state["teams"]:
        raise HTTPException(status_code=422, detail=f"Unknown team: {req.home_team}")
    if req.away_team not in _state["teams"]:
        raise HTTPException(status_code=422, detail=f"Unknown team: {req.away_team}")
    if req.home_team == req.away_team:
        raise HTTPException(status_code=422, detail="Home and away teams must be different")

    features = build_features(
        df=_state["df"],
        home_team=req.home_team,
        away_team=req.away_team,
        match_date=pd.Timestamp.now(),
        neutral=req.neutral_venue,
    )

    X = pd.DataFrame([features])
    probs = _state["predict_fn"](X)

    return PredictionResponse(
        home_win=round(float(probs["home_win"].iloc[0]), 4),
        draw=round(float(probs["draw"].iloc[0]), 4),
        away_win=round(float(probs["away_win"].iloc[0]), 4),
        model_used=_state["model_type"],
    )
