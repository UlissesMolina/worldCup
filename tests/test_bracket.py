import pandas as pd
import pytest

from src.simulation.bracket import (
    estimate_scoreline,
    compute_group_standings,
    simulate_group_stage,
    rank_third_place,
    seed_bracket,
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
