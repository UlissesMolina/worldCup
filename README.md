# World Cup Match Predictor

Full-stack prediction system for the 2026 FIFA World Cup. Trains XGBoost and PyTorch models on 45,000+ historical international matches, serves live win-probability predictions via FastAPI, and runs Monte Carlo tournament simulations across all 48 teams.

## Features

- **Match Prediction** — predict win/draw/loss probabilities for any international matchup
- **Monte Carlo Simulation** — simulate the entire 2026 World Cup 10,000 times to estimate each team's championship probability
- **Results Comparison** — compare model predictions against actual 2026 World Cup match results with per-match accuracy tracking
- **Feature Engineering** — team form, FIFA ranking differentials, head-to-head history, rest days, goal difference trends, and tournament encoding
- **Model Comparison** — trains both XGBoost and PyTorch neural network, selects the best on a date-based holdout set
- **Web Frontend** — interactive UI with match predictor and full tournament simulation results

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
- `--min-year 1990` — earliest year to include in training data
- `--epochs 100` — PyTorch training epochs

The pipeline downloads data, engineers features, trains both models, evaluates with per-class precision/recall/F1 and confusion matrix, then saves the winner to `models/`.

## Running the API

```bash
uvicorn src.api.main:app --reload
```

Open `http://localhost:8000` for the web interface.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web frontend |
| `GET` | `/teams` | List all available teams |
| `POST` | `/predict` | Predict a single match outcome |
| `POST` | `/simulate-bracket` | Run Monte Carlo tournament simulation |
| `GET` | `/results-comparison` | Compare predictions vs actual results |

### Predict a match

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Brazil", "away_team": "Germany", "neutral_venue": false}'
```

### Simulate the tournament

```bash
curl -X POST http://localhost:8000/simulate-bracket \
  -H "Content-Type: application/json" \
  -d '{"n_simulations": 10000}'
```

Returns each team's probability of advancing through group stage, R32, R16, quarterfinals, semifinals, final, and winning the championship.

### Compare predictions vs actual results

```bash
curl http://localhost:8000/results-comparison
```

Returns each match's predicted probabilities alongside actual scores, with per-group and overall accuracy percentages.

## Project Structure

```
src/
├── api/            # FastAPI app, schemas, Lambda handler
├── data/           # Data download, processing, 2026 group draw
├── features/       # Feature engineering pipeline
├── models/         # XGBoost and PyTorch model training, evaluation
└── simulation/     # Group stage, knockout, and Monte Carlo simulation
scripts/
└── train.py        # Training CLI
static/
└── index.html      # Web frontend
models/
└── model.pkl       # Trained model artifact
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
