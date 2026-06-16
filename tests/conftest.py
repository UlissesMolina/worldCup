import pandas as pd
import pytest


@pytest.fixture
def sample_matches():
    """Minimal match dataset for testing."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2020-01-10", "2020-03-15", "2020-06-20",
            "2021-01-05", "2021-03-10", "2021-06-15",
            "2022-01-08", "2022-06-12", "2023-01-14",
            "2023-06-10",
        ]),
        "home_team": [
            "Brazil", "Germany", "Brazil", "Argentina", "Brazil",
            "Germany", "Brazil", "Argentina", "Brazil", "Germany",
        ],
        "away_team": [
            "Argentina", "Brazil", "Germany", "Brazil", "Argentina",
            "Brazil", "Argentina", "Germany", "Germany", "Brazil",
        ],
        "home_score": [2, 1, 3, 0, 1, 2, 2, 1, 0, 3],
        "away_score": [1, 1, 0, 0, 1, 3, 0, 2, 0, 1],
        "tournament": [
            "Friendly", "FIFA World Cup qualification", "FIFA World Cup",
            "Friendly", "Copa América", "FIFA World Cup qualification",
            "FIFA World Cup", "Copa América", "Friendly", "FIFA World Cup",
        ],
        "city": ["São Paulo"] * 10,
        "country": ["Brazil"] * 10,
        "neutral": [False, False, False, False, False, False, True, False, False, True],
    })
