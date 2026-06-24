"""CLI entrypoint: download → process → features → train → evaluate → save winner."""
import argparse
import sys
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import classification_report as sk_report

from src.data.download import download_match_data
from src.data.process import clean_matches, add_target, process_and_save
from src.features.engineering import build_dataset
from src.models.xgboost_model import train_xgboost, predict_xgboost, save_xgboost
from src.models.pytorch_model import train_pytorch, predict_pytorch, save_pytorch
from src.models.evaluate import evaluate_model, select_best_model


ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


def main():
    parser = argparse.ArgumentParser(description="Train World Cup predictor models")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download if CSVs already exist")
    parser.add_argument("--holdout-year", type=int, default=2022, help="Year to split train/test")
    parser.add_argument("--min-year", type=int, default=1990, help="Earliest year to include")
    parser.add_argument("--epochs", type=int, default=100, help="PyTorch training epochs")
    args = parser.parse_args()

    # 1. Download
    if not args.skip_download:
        print("Downloading match data...")
        download_match_data(DATA_RAW)
    else:
        print("Skipping download.")

    # 2. Process
    print("Processing data...")
    parquet_path = process_and_save(DATA_RAW, DATA_PROCESSED, min_year=args.min_year)
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} matches loaded.")

    # 3. Build features
    print("Building feature matrix (this may take a while)...")
    X, y = build_dataset(df)
    print(f"  Feature matrix: {X.shape}")

    # 4. Date-based split
    split_date = pd.Timestamp(f"{args.holdout_year}-01-01")
    train_mask = df["date"] < split_date
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 5. Train XGBoost
    print("Training XGBoost...")
    xgb_trained = train_xgboost(X_train, y_train)
    xgb_probs = predict_xgboost(xgb_trained, X_test)
    xgb_metrics = evaluate_model(y_test, xgb_probs)
    print(f"  XGBoost — log_loss: {xgb_metrics['log_loss']:.4f}, accuracy: {xgb_metrics['accuracy']:.4f}")

    print("\n  XGBoost per-class report:")
    xgb_pred_labels = xgb_probs.idxmax(axis=1)
    print(sk_report(
        y_test, xgb_pred_labels,
        target_names=["away_win", "draw", "home_win"],
        labels=["away_win", "draw", "home_win"],
        zero_division=0,
    ))

    # 6. Train PyTorch
    print(f"Training PyTorch ({args.epochs} epochs)...")
    pt_trained = train_pytorch(X_train, y_train, epochs=args.epochs)
    pt_probs = predict_pytorch(pt_trained, X_test)
    pt_metrics = evaluate_model(y_test, pt_probs)
    print(f"  PyTorch — log_loss: {pt_metrics['log_loss']:.4f}, accuracy: {pt_metrics['accuracy']:.4f}")

    print("\n  PyTorch per-class report:")
    pt_pred_labels = pt_probs.idxmax(axis=1)
    print(sk_report(
        y_test, pt_pred_labels,
        target_names=["away_win", "draw", "home_win"],
        labels=["away_win", "draw", "home_win"],
        zero_division=0,
    ))

    # 7. Select winner
    results = {"xgboost": xgb_metrics, "pytorch": pt_metrics}
    winner = select_best_model(results)
    print(f"\nWinner: {winner}")

    # 8. Save winner
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if winner == "xgboost":
        save_xgboost(xgb_trained, MODELS_DIR / "model.pkl")
    else:
        save_pytorch(pt_trained, MODELS_DIR / "model.pkl")

    # Save metadata
    meta_path = MODELS_DIR / "metadata.txt"
    meta_path.write_text(f"model_type={winner}\n")
    print(f"Model saved to {MODELS_DIR / 'model.pkl'}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
