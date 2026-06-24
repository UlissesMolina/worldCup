import random

import pandas as pd
import pytest

from src.simulation.bracket import (
    estimate_scoreline,
    compute_group_standings,
    simulate_group_stage,
    rank_third_place,
    seed_bracket,
    simulate_knockout_round,
    simulate_tournament,
    sample_match_outcome,
    simulate_tournament_monte_carlo,
)


def test_estimate_scoreline_home_win():
    home, away = estimate_scoreline("home_win", 0.7, 0.1, 0.2)
    assert home > away
    assert isinstance(home, int)
    assert isinstance(away, int)


def test_estimate_scoreline_draw():
    home, away = estimate_scoreline("draw", 0.3, 0.5, 0.2)
    assert home == away


def test_estimate_scoreline_away_win():
    home, away = estimate_scoreline("away_win", 0.1, 0.2, 0.7)
    assert away > home


def test_compute_group_standings():
    matches = [
        {"home": "Brazil", "away": "Morocco", "result": "home_win",
         "home_goals": 2, "away_goals": 0, "home_win": 0.6, "draw": 0.2, "away_win": 0.2},
        {"home": "Scotland", "away": "Haiti", "result": "draw",
         "home_goals": 1, "away_goals": 1, "home_win": 0.4, "draw": 0.3, "away_win": 0.3},
        {"home": "Brazil", "away": "Haiti", "result": "home_win",
         "home_goals": 3, "away_goals": 0, "home_win": 0.8, "draw": 0.1, "away_win": 0.1},
        {"home": "Morocco", "away": "Scotland", "result": "home_win",
         "home_goals": 1, "away_goals": 0, "home_win": 0.5, "draw": 0.3, "away_win": 0.2},
        {"home": "Brazil", "away": "Scotland", "result": "home_win",
         "home_goals": 2, "away_goals": 1, "home_win": 0.55, "draw": 0.25, "away_win": 0.2},
        {"home": "Haiti", "away": "Morocco", "result": "away_win",
         "home_goals": 0, "away_goals": 2, "home_win": 0.1, "draw": 0.2, "away_win": 0.7},
    ]
    standings = compute_group_standings(
        ["Brazil", "Morocco", "Scotland", "Haiti"], matches
    )
    assert standings[0]["team"] == "Brazil"
    assert standings[0]["pts"] == 9
    assert standings[1]["team"] == "Morocco"
    assert standings[1]["pts"] == 6
    assert standings[0]["pos"] == 1
    assert standings[1]["pos"] == 2


def test_rank_third_place():
    group_results = {}
    third_place_data = [
        ("A", {"team": "T_A3", "pts": 4, "gd": 2, "gf": 3, "pos": 3}),
        ("B", {"team": "T_B3", "pts": 3, "gd": 1, "gf": 2, "pos": 3}),
        ("C", {"team": "T_C3", "pts": 4, "gd": 3, "gf": 4, "pos": 3}),
        ("D", {"team": "T_D3", "pts": 2, "gd": -1, "gf": 1, "pos": 3}),
        ("E", {"team": "T_E3", "pts": 3, "gd": 0, "gf": 2, "pos": 3}),
        ("F", {"team": "T_F3", "pts": 1, "gd": -3, "gf": 0, "pos": 3}),
        ("G", {"team": "T_G3", "pts": 4, "gd": 1, "gf": 3, "pos": 3}),
        ("H", {"team": "T_H3", "pts": 3, "gd": 2, "gf": 3, "pos": 3}),
        ("I", {"team": "T_I3", "pts": 2, "gd": 0, "gf": 1, "pos": 3}),
        ("J", {"team": "T_J3", "pts": 1, "gd": -2, "gf": 1, "pos": 3}),
        ("K", {"team": "T_K3", "pts": 3, "gd": 1, "gf": 2, "pos": 3}),
        ("L", {"team": "T_L3", "pts": 2, "gd": -1, "gf": 1, "pos": 3}),
    ]
    for letter, standing in third_place_data:
        group_results[letter] = {
            "standings": [
                {"team": f"T_{letter}1", "pts": 7, "gd": 5, "gf": 6, "pos": 1},
                {"team": f"T_{letter}2", "pts": 5, "gd": 3, "gf": 4, "pos": 2},
                standing,
                {"team": f"T_{letter}4", "pts": 0, "gd": -5, "gf": 0, "pos": 4},
            ],
            "matches": [],
        }

    ranked = rank_third_place(group_results)
    assert len(ranked) == 12
    advanced = [r for r in ranked if r["advanced"]]
    eliminated = [r for r in ranked if not r["advanced"]]
    assert len(advanced) == 8
    assert len(eliminated) == 4
    assert ranked[0]["pts"] == 4


def test_seed_bracket():
    group_results = {}
    for letter in "ABCDEFGHIJKL":
        group_results[letter] = {
            "standings": [
                {"team": f"1{letter}", "pos": 1, "pts": 7, "gd": 5, "gf": 6, "advanced": True},
                {"team": f"2{letter}", "pos": 2, "pts": 4, "gd": 2, "gf": 3, "advanced": True},
                {"team": f"3{letter}", "pos": 3, "pts": 2, "gd": 0, "gf": 1, "advanced": False},
                {"team": f"4{letter}", "pos": 4, "pts": 0, "gd": -5, "gf": 0, "advanced": False},
            ],
            "matches": [],
        }
    advancing_groups = ["A", "B", "C", "E", "F", "G", "H", "I"]
    third_place_ranked = []
    for letter in "ABCDEFGHIJKL":
        entry = {
            "team": f"3{letter}",
            "group": letter,
            "pts": 3 if letter in advancing_groups else 0,
            "gd": 0, "gf": 1,
            "advanced": letter in advancing_groups,
        }
        third_place_ranked.append(entry)

    r32 = seed_bracket(group_results, third_place_ranked)
    assert len(r32) == 16

    for match in r32:
        home_group = None
        away_group = None
        for letter, gr in group_results.items():
            for s in gr["standings"]:
                if s["team"] == match["home"]:
                    home_group = letter
                if s["team"] == match["away"]:
                    away_group = letter
        if home_group and away_group:
            assert home_group != away_group, f"{match['home']} vs {match['away']} are from same group {home_group}"


def test_sample_match_outcome_respects_distribution():
    random.seed(42)
    probs = {"home_win": 0.6, "draw": 0.2, "away_win": 0.2}
    counts = {"home_win": 0, "draw": 0, "away_win": 0}
    n = 10000
    for _ in range(n):
        outcome = sample_match_outcome(probs)
        counts[outcome] += 1

    assert abs(counts["home_win"] / n - 0.6) < 0.05
    assert abs(counts["draw"] / n - 0.2) < 0.05
    assert abs(counts["away_win"] / n - 0.2) < 0.05


def test_sample_match_outcome_knockout_no_draw():
    random.seed(42)
    probs = {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}
    for _ in range(1000):
        outcome = sample_match_outcome(probs, allow_draw=False)
        assert outcome in ("home_win", "away_win")


def _make_mock_predict_fn():
    """Return a predict_fn that always gives home team 60% win."""
    def predict_fn(X):
        return pd.DataFrame(
            [{"away_win": 0.15, "draw": 0.25, "home_win": 0.60}]
        )
    return predict_fn


def test_simulate_knockout_round():
    matchups = [
        {"home": "Brazil", "away": "Germany"},
        {"home": "France", "away": "Spain"},
    ]
    fake_df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "home_team": ["Brazil", "France"],
        "away_team": ["Germany", "Spain"],
        "home_score": [2, 1],
        "away_score": [1, 0],
        "tournament": ["Friendly", "Friendly"],
        "neutral": [False, False],
    })
    from src.features.engineering import compute_elo_ratings
    elo = compute_elo_ratings(fake_df)
    predict_fn = _make_mock_predict_fn()

    results = simulate_knockout_round(
        matchups, fake_df, elo, predict_fn, pd.Timestamp("2025-06-15")
    )
    assert len(results) == 2
    assert results[0]["winner"] == "Brazil"
    assert results[1]["winner"] == "France"


def test_simulate_tournament_returns_full_structure():
    fake_df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        "home_team": ["Brazil", "Germany", "France"],
        "away_team": ["Germany", "France", "Brazil"],
        "home_score": [2, 1, 0],
        "away_score": [1, 0, 0],
        "tournament": ["Friendly", "Friendly", "Friendly"],
        "neutral": [False, False, False],
    })
    from src.features.engineering import compute_elo_ratings
    elo = compute_elo_ratings(fake_df)
    predict_fn = _make_mock_predict_fn()

    result = simulate_tournament(fake_df, elo, predict_fn)

    assert "groups" in result
    assert len(result["groups"]) == 12
    assert "knockout" in result
    assert "r32" in result["knockout"]
    assert len(result["knockout"]["r32"]) == 16
    assert "r16" in result["knockout"]
    assert len(result["knockout"]["r16"]) == 8
    assert "qf" in result["knockout"]
    assert len(result["knockout"]["qf"]) == 4
    assert "sf" in result["knockout"]
    assert len(result["knockout"]["sf"]) == 2
    assert "final" in result["knockout"]
    assert "champion" in result


def test_simulate_tournament_monte_carlo_shape():
    fake_df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        "home_team": ["Brazil", "Germany", "France"],
        "away_team": ["Germany", "France", "Brazil"],
        "home_score": [2, 1, 0],
        "away_score": [1, 0, 0],
        "tournament": ["Friendly", "Friendly", "Friendly"],
        "neutral": [False, False, False],
    })
    from src.features.engineering import compute_elo_ratings
    elo = compute_elo_ratings(fake_df)

    def predict_fn(X):
        return pd.DataFrame(
            [{"away_win": 0.15, "draw": 0.25, "home_win": 0.60}]
        )

    result = simulate_tournament_monte_carlo(
        fake_df, elo, predict_fn, n_simulations=50
    )

    assert "teams" in result
    assert "simulations" in result
    assert result["simulations"] == 50

    # Check that all 48 World Cup teams have entries
    assert len(result["teams"]) == 48

    # Check shape of each team's data
    stages = ["group_pct", "r32_pct", "r16_pct", "qf_pct", "sf_pct", "final_pct", "champion_pct"]
    for team, data in result["teams"].items():
        for stage in stages:
            assert stage in data, f"Missing {stage} for {team}"
            assert 0 <= data[stage] <= 100, f"{stage} for {team} is {data[stage]}"

    # At least one team should have champion_pct > 0
    champion_total = sum(t["champion_pct"] for t in result["teams"].values())
    assert abs(champion_total - 100.0) < 1.0  # should sum to ~100%
