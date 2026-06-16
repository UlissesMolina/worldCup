from pydantic import BaseModel


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    neutral_venue: bool = False


class PredictionResponse(BaseModel):
    home_win: float
    draw: float
    away_win: float
    model_used: str
