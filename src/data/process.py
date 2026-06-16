import pandas as pd
from pathlib import Path


def clean_matches(df: pd.DataFrame, min_year: int = 1990) -> pd.DataFrame:
    """Filter matches to min_year+ and sort by date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= min_year]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add target column: home_win, draw, or away_win."""
    df = df.copy()

    def _outcome(row):
        if row["home_score"] > row["away_score"]:
            return "home_win"
        elif row["home_score"] < row["away_score"]:
            return "away_win"
        else:
            return "draw"

    df["target"] = df.apply(_outcome, axis=1)
    return df


def process_and_save(raw_dir: Path, output_dir: Path, min_year: int = 1990) -> Path:
    """Full processing pipeline: load CSV, clean, add target, save parquet."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_dir / "results.csv")
    df = clean_matches(df, min_year=min_year)
    df = add_target(df)

    output_path = output_dir / "matches.parquet"
    df.to_parquet(output_path, index=False)
    return output_path
