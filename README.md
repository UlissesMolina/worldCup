# World Cup Match Predictor

Predicts win/draw/loss probabilities for international football matches using XGBoost and PyTorch models trained on 45,000+ historical results.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements-dev.txt
```

## Training

```bash
python scripts/train.py
```

Options:
- `--skip-download` — skip Kaggle download if data already exists
- `--holdout-year 2022` — year to split train/test
- `--epochs 100` — PyTorch training epochs

## Running the API

```bash
uvicorn src.api.main:app --reload
```

## Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Brazil", "away_team": "Germany", "neutral_venue": false}'
```

## Tests

```bash
pytest -v
```

## Deployment (AWS Lambda)

```bash
docker build -t worldcup-predictor .
# Tag and push to ECR, then create/update Lambda function
```
