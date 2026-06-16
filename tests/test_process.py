import pandas as pd
from src.data.process import clean_matches, add_target


def test_clean_matches_filters_by_year(sample_matches):
    result = clean_matches(sample_matches, min_year=2021)
    assert result["date"].dt.year.min() >= 2021
    assert len(result) < len(sample_matches)


def test_clean_matches_sorts_by_date(sample_matches):
    result = clean_matches(sample_matches, min_year=2020)
    dates = result["date"].tolist()
    assert dates == sorted(dates)


def test_add_target_home_win(sample_matches):
    result = add_target(sample_matches)
    # First row: Brazil 2 - Argentina 1 → home_win
    assert result.iloc[0]["target"] == "home_win"


def test_add_target_draw(sample_matches):
    result = add_target(sample_matches)
    # Second row: Germany 1 - Brazil 1 → draw
    assert result.iloc[1]["target"] == "draw"


def test_add_target_away_win(sample_matches):
    result = add_target(sample_matches)
    # Row index 5: Germany 2 - Brazil 3 → away_win
    assert result.iloc[5]["target"] == "away_win"


def test_add_target_column_values(sample_matches):
    result = add_target(sample_matches)
    assert set(result["target"].unique()).issubset({"home_win", "draw", "away_win"})
