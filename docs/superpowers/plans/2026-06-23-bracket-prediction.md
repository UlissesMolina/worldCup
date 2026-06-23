# Tournament Bracket Prediction Tab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Tournament Bracket" tab that simulates the entire 2026 FIFA World Cup and displays a visual bracket from group stage through the final.

**Architecture:** New `src/simulation/bracket.py` module handles all tournament logic (group simulation, standings, seeding, knockout). New `POST /simulate-bracket` endpoint in FastAPI returns the full bracket as JSON. Frontend adds a tab with group tables and a classic bracket tree, all vanilla JS + Tailwind CSS.

**Tech Stack:** Python, pandas, FastAPI, vanilla JavaScript, Tailwind CSS

---

## File Structure

**New files:**
- `src/data/groups.py` — 2026 World Cup group draw (12 groups × 4 teams)
- `src/simulation/__init__.py` — empty package init
- `src/simulation/bracket.py` — tournament simulation logic
- `tests/test_bracket.py` — simulation unit tests

**Modified files:**
- `src/api/main.py` — add `/simulate-bracket` endpoint
- `static/index.html` — add tab system + bracket tab UI
- `tests/test_api.py` — add bracket endpoint test

---

### Task 1: Create World Cup Groups Data

**Files:**
- Create: `src/data/groups.py`

- [ ] **Step 1: Create the groups data file**

```python
# src/data/groups.py
"""2026 FIFA World Cup group draw — 48 teams in 12 groups of 4."""

# Team names match the historical dataset (data/processed/matches.parquet).
# Mapping notes:
#   FIFA "Czechia" → dataset "Czech Republic"
#   FIFA "Türkiye" → dataset "Turkey"
#   FIFA "Curaçao" → dataset "Curacao" (may not exist; use as-is)
#   FIFA "Cabo Verde" → dataset "Cape Verde"
#   FIFA "Côte d'Ivoire" → dataset "Ivory Coast"
#   FIFA "Bosnia-Herzegovina" → dataset "Bosnia and Herzegovina"

GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# Fixed R32 pairings (runner-up vs runner-up, and winner vs runner-up).
# These 8 matches don't involve third-place teams.
FIXED_R32 = [
    ("2A", "2B"),  # Match 73
    ("1C", "2F"),  # Match 76
    ("1F", "2C"),  # Match 75
    ("2E", "2I"),  # Match 78
    ("1H", "2J"),  # Match 84
    ("1J", "2H"),  # Match 86 (Argentina side)
    ("2K", "2L"),  # Match 83
    ("2D", "2G"),  # Match 88
]

# Variable R32 pairings: group winners vs third-place teams.
# Each group winner has a pool of valid third-place opponents (no same-group clash).
# Key = group winner's group letter, Value = list of valid 3rd-place source groups.
WINNER_VS_THIRD_POOLS = {
    "A": ["C", "E", "F", "H", "I"],       # Match 79 (Mexico side)
    "B": ["E", "F", "G", "I", "J"],       # Match 85
    "D": ["B", "E", "F", "I", "J"],       # Match 81 (USA side)
    "E": ["A", "B", "C", "D", "F"],       # Match 74 (Germany side)
    "G": ["A", "E", "H", "I", "J"],       # Match 82
    "I": ["C", "D", "F", "G", "H"],       # Match 77
    "K": ["D", "E", "I", "J", "L"],       # Match 87
    "L": ["E", "H", "I", "J", "K"],       # Match 80
}

# R32 match ordering for bracket layout (left half then right half).
# Each entry is an index into the final R32 matchups list.
# Left bracket: matches 0-7, Right bracket: matches 8-15
# R16 pairings: winner of match 0 vs winner of match 1, etc.
# QF: winner of R16-0 vs winner of R16-1, etc.
# SF: winner of QF-0 vs winner of QF-1, etc.
# Final: winner of SF-0 vs winner of SF-1
BRACKET_STRUCTURE = {
    "r32_left": [0, 1, 2, 3, 4, 5, 6, 7],
    "r32_right": [8, 9, 10, 11, 12, 13, 14, 15],
    "r16": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
    "qf": [(0, 1), (2, 3), (4, 5), (6, 7)],
    "sf": [(0, 1), (2, 3)],
    "final": (0, 1),
}
```

- [ ] **Step 2: Commit**

```bash
git add src/data/groups.py
git commit -m "feat: add 2026 World Cup group draw data"
```

---

### Task 2: Group Stage Simulation Logic

**Files:**
- Create: `src/simulation/__init__.py`
- Create: `src/simulation/bracket.py`
- Create: `tests/test_bracket.py`

- [ ] **Step 1: Create empty package init**

```python
# src/simulation/__init__.py
```

- [ ] **Step 2: Write failing tests for group stage helpers**

```python
# tests/test_bracket.py
import pandas as pd
import pytest

from src.simulation.bracket import (
    estimate_scoreline,
    compute_group_standings,
    simulate_group_stage,
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
    # Brazil: 3 wins = 9 pts, Morocco: 2 wins = 6 pts
    assert standings[0]["team"] == "Brazil"
    assert standings[0]["pts"] == 9
    assert standings[1]["team"] == "Morocco"
    assert standings[1]["pts"] == 6
    assert standings[0]["pos"] == 1
    assert standings[1]["pos"] == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_bracket.py -v`
Expected: FAIL with ImportError (module doesn't exist yet)

- [ ] **Step 4: Implement group stage helpers**

```python
# src/simulation/bracket.py
"""Tournament simulation: group stage, standings, seeding, knockout."""

import pandas as pd

from src.features.engineering import build_features


def estimate_scoreline(
    result: str, home_win_prob: float, draw_prob: float, away_win_prob: float
) -> tuple[int, int]:
    """Estimate a scoreline from the predicted result and probabilities.

    Returns (home_goals, away_goals). Uses probability magnitude to scale margin.
    """
    if result == "home_win":
        margin = max(1, round(home_win_prob * 3))
        home = 1 + margin
        away = max(0, home - margin)
        return (home, away)
    elif result == "away_win":
        margin = max(1, round(away_win_prob * 3))
        away = 1 + margin
        home = max(0, away - margin)
        return (home, away)
    else:  # draw
        goals = max(0, round(draw_prob * 2))
        return (goals, goals)


def predict_match(
    home_team: str,
    away_team: str,
    df: pd.DataFrame,
    elo_ratings: dict,
    predict_fn,
    match_date: pd.Timestamp,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
) -> dict:
    """Predict a single match. Returns dict with teams, probs, result, goals."""
    features = build_features(
        df=df,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        neutral=neutral,
        tournament=tournament,
        elo_ratings=elo_ratings,
    )
    X = pd.DataFrame([features])
    probs = predict_fn(X)
    home_win_prob = float(probs["home_win"].iloc[0])
    draw_prob = float(probs["draw"].iloc[0])
    away_win_prob = float(probs["away_win"].iloc[0])

    # Determine result: highest probability wins
    prob_map = {"home_win": home_win_prob, "draw": draw_prob, "away_win": away_win_prob}
    result = max(prob_map, key=prob_map.get)

    home_goals, away_goals = estimate_scoreline(result, home_win_prob, draw_prob, away_win_prob)

    return {
        "home": home_team,
        "away": away_team,
        "home_win": round(home_win_prob, 4),
        "draw": round(draw_prob, 4),
        "away_win": round(away_win_prob, 4),
        "result": result,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def compute_group_standings(teams: list[str], matches: list[dict]) -> list[dict]:
    """Compute group standings from match results.

    Ranking: points → goal difference → goals scored → alphabetical.
    Returns list of dicts sorted by rank, each with pos, team, w, d, l, gf, ga, gd, pts.
    """
    stats = {t: {"w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0} for t in teams}

    for m in matches:
        h, a = m["home"], m["away"]
        hg, ag = m["home_goals"], m["away_goals"]
        stats[h]["gf"] += hg
        stats[h]["ga"] += ag
        stats[a]["gf"] += ag
        stats[a]["ga"] += hg

        if m["result"] == "home_win":
            stats[h]["w"] += 1
            stats[a]["l"] += 1
        elif m["result"] == "away_win":
            stats[a]["w"] += 1
            stats[h]["l"] += 1
        else:
            stats[h]["d"] += 1
            stats[a]["d"] += 1

    rows = []
    for team, s in stats.items():
        pts = s["w"] * 3 + s["d"]
        gd = s["gf"] - s["ga"]
        rows.append({
            "team": team,
            "w": s["w"], "d": s["d"], "l": s["l"],
            "gf": s["gf"], "ga": s["ga"], "gd": gd, "pts": pts,
        })

    # Sort: pts desc, gd desc, gf desc, team asc (alphabetical tiebreak)
    rows.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], r["team"]))

    for i, row in enumerate(rows):
        row["pos"] = i + 1

    return rows


def simulate_group_stage(
    groups: dict[str, list[str]],
    df: pd.DataFrame,
    elo_ratings: dict,
    predict_fn,
    match_date: pd.Timestamp,
) -> dict:
    """Simulate all group stage matches.

    Returns dict keyed by group letter, each with 'standings' and 'matches'.
    """
    from itertools import combinations

    result = {}
    for group_letter, teams in groups.items():
        matches = []
        # Round-robin: each pair plays once. First-listed team is "home".
        for home, away in combinations(teams, 2):
            match = predict_match(
                home_team=home,
                away_team=away,
                df=df,
                elo_ratings=elo_ratings,
                predict_fn=predict_fn,
                match_date=match_date,
                neutral=True,
                tournament="FIFA World Cup",
            )
            matches.append(match)

        standings = compute_group_standings(teams, matches)
        # Mark who advances: pos 1 and 2 always advance
        for row in standings:
            row["advanced"] = row["pos"] <= 2

        result[group_letter] = {
            "standings": standings,
            "matches": matches,
        }

    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_bracket.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/simulation/__init__.py src/simulation/bracket.py tests/test_bracket.py
git commit -m "feat: add group stage simulation logic with standings"
```

---

### Task 3: Third-Place Ranking and Bracket Seeding

**Files:**
- Modify: `src/simulation/bracket.py`
- Modify: `tests/test_bracket.py`

- [ ] **Step 1: Write failing tests for third-place ranking and seeding**

Add to `tests/test_bracket.py`:

```python
from src.simulation.bracket import rank_third_place, seed_bracket


def test_rank_third_place():
    # Simulate 12 group standings where position 3 = third-place team
    group_results = {}
    # Create 12 groups with varying third-place stats
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
    # Top 8 should advance
    advanced = [r for r in ranked if r["advanced"]]
    eliminated = [r for r in ranked if not r["advanced"]]
    assert len(advanced) == 8
    assert len(eliminated) == 4
    # Best third-place teams (pts 4) should be at top
    assert ranked[0]["pts"] == 4


def test_seed_bracket():
    # Build minimal group_results with known positions
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
    # Mark 8 third-place teams as advancing
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

    # Check no team plays a team from their own group
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bracket.py::test_rank_third_place tests/test_bracket.py::test_seed_bracket -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement third-place ranking and bracket seeding**

Add to `src/simulation/bracket.py`:

```python
from src.data.groups import FIXED_R32, WINNER_VS_THIRD_POOLS


def rank_third_place(group_results: dict) -> list[dict]:
    """Rank all 12 third-place teams. Top 8 advance.

    Returns list of 12 dicts sorted by rank, each with 'advanced' flag.
    """
    thirds = []
    for group_letter, data in group_results.items():
        for row in data["standings"]:
            if row["pos"] == 3:
                entry = {**row, "group": group_letter}
                thirds.append(entry)
                break

    # Sort: pts desc, gd desc, gf desc, team asc
    thirds.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], r["team"]))

    for i, entry in enumerate(thirds):
        entry["advanced"] = i < 8

    return thirds


def _get_team_by_position(group_results: dict, group: str, pos: int) -> str:
    """Get team name by group letter and position (1, 2, 3)."""
    for row in group_results[group]["standings"]:
        if row["pos"] == pos:
            return row["team"]
    raise ValueError(f"No team at position {pos} in group {group}")


def seed_bracket(
    group_results: dict, third_place_ranked: list[dict]
) -> list[dict]:
    """Create 16 R32 matchups from group results and third-place ranking.

    Returns list of 16 dicts, each with 'home' and 'away' team names.
    Order matters: indices 0-7 are left bracket, 8-15 are right bracket.
    R16 pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15).
    """
    # Build the 8 fixed matches (runner-up vs runner-up, winner vs runner-up)
    fixed_matches = []
    for home_code, away_code in FIXED_R32:
        home_pos = int(home_code[0])
        home_group = home_code[1]
        away_pos = int(away_code[0])
        away_group = away_code[1]
        home_team = _get_team_by_position(group_results, home_group, home_pos)
        away_team = _get_team_by_position(group_results, away_group, away_pos)
        fixed_matches.append({"home": home_team, "away": away_team})

    # Build the 8 winner-vs-third matches
    advancing_thirds = {
        entry["group"]: entry["team"]
        for entry in third_place_ranked
        if entry["advanced"]
    }
    assigned_groups = set()

    winner_matches = []
    # Process winners in order: A, B, D, E, G, I, K, L
    for winner_group in sorted(WINNER_VS_THIRD_POOLS.keys()):
        pool = WINNER_VS_THIRD_POOLS[winner_group]
        winner_team = _get_team_by_position(group_results, winner_group, 1)
        # Pick first available third-place team from valid pool
        third_team = None
        for candidate_group in pool:
            if candidate_group in advancing_thirds and candidate_group not in assigned_groups:
                third_team = advancing_thirds[candidate_group]
                assigned_groups.add(candidate_group)
                break
        if third_team is None:
            # Fallback: pick any remaining advancing third
            for g, t in advancing_thirds.items():
                if g not in assigned_groups:
                    third_team = t
                    assigned_groups.add(g)
                    break
        winner_matches.append({"home": winner_team, "away": third_team})

    # Interleave into bracket order:
    # Left bracket (0-7): fixed[0], winner[0], fixed[1], winner[1], fixed[2], winner[2], fixed[3], winner[3]
    # Right bracket (8-15): winner[4], fixed[4], winner[5], fixed[5], winner[6], fixed[6], winner[7], fixed[7]
    r32 = []
    for i in range(4):
        r32.append(fixed_matches[i])
        r32.append(winner_matches[i])
    for i in range(4):
        r32.append(winner_matches[4 + i])
        r32.append(fixed_matches[4 + i])

    return r32
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bracket.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/simulation/bracket.py tests/test_bracket.py
git commit -m "feat: add third-place ranking and bracket seeding"
```

---

### Task 4: Knockout Simulation and Tournament Orchestrator

**Files:**
- Modify: `src/simulation/bracket.py`
- Modify: `tests/test_bracket.py`

- [ ] **Step 1: Write failing tests for knockout and tournament orchestrator**

Add to `tests/test_bracket.py`:

```python
from src.simulation.bracket import simulate_knockout_round, simulate_tournament


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
    # With 60% home_win, home team should win
    assert results[0]["winner"] == "Brazil"
    assert results[1]["winner"] == "France"


def test_simulate_tournament_returns_full_structure():
    """Smoke test: simulate_tournament returns expected keys and shapes."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bracket.py::test_simulate_knockout_round tests/test_bracket.py::test_simulate_tournament_returns_full_structure -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement knockout simulation and tournament orchestrator**

Add to `src/simulation/bracket.py`:

```python
def simulate_knockout_round(
    matchups: list[dict],
    df: pd.DataFrame,
    elo_ratings: dict,
    predict_fn,
    match_date: pd.Timestamp,
) -> list[dict]:
    """Simulate a knockout round. Each matchup dict has 'home' and 'away'.

    Returns list of match result dicts, each with 'winner' added.
    In knockouts, if draw is most likely, team with higher win prob advances.
    """
    results = []
    for matchup in matchups:
        match = predict_match(
            home_team=matchup["home"],
            away_team=matchup["away"],
            df=df,
            elo_ratings=elo_ratings,
            predict_fn=predict_fn,
            match_date=match_date,
            neutral=True,
            tournament="FIFA World Cup",
        )
        # Determine winner: no draws in knockouts
        if match["result"] == "home_win":
            match["winner"] = match["home"]
        elif match["result"] == "away_win":
            match["winner"] = match["away"]
        else:
            # Draw: team with higher win probability advances
            if match["home_win"] >= match["away_win"]:
                match["winner"] = match["home"]
            else:
                match["winner"] = match["away"]
        results.append(match)
    return results


def simulate_tournament(
    df: pd.DataFrame,
    elo_ratings: dict,
    predict_fn,
    match_date: pd.Timestamp | None = None,
) -> dict:
    """Simulate the full 2026 World Cup tournament.

    Returns the complete bracket: groups, knockout rounds, and champion.
    """
    from src.data.groups import GROUPS

    if match_date is None:
        match_date = pd.Timestamp("2026-06-11")

    # 1. Group stage
    group_results = simulate_group_stage(
        GROUPS, df, elo_ratings, predict_fn, match_date
    )

    # 2. Third-place ranking
    third_place = rank_third_place(group_results)

    # Update group standings with third-place advancement
    for entry in third_place:
        if entry["advanced"]:
            group = entry["group"]
            for row in group_results[group]["standings"]:
                if row["pos"] == 3:
                    row["advanced"] = True

    # 3. Seed bracket
    r32_matchups = seed_bracket(group_results, third_place)

    # 4. Knockout rounds
    r32 = simulate_knockout_round(r32_matchups, df, elo_ratings, predict_fn, match_date)

    r16_matchups = [
        {"home": r32[i]["winner"], "away": r32[i + 1]["winner"]}
        for i in range(0, 16, 2)
    ]
    r16 = simulate_knockout_round(r16_matchups, df, elo_ratings, predict_fn, match_date)

    qf_matchups = [
        {"home": r16[i]["winner"], "away": r16[i + 1]["winner"]}
        for i in range(0, 8, 2)
    ]
    qf = simulate_knockout_round(qf_matchups, df, elo_ratings, predict_fn, match_date)

    sf_matchups = [
        {"home": qf[i]["winner"], "away": qf[i + 1]["winner"]}
        for i in range(0, 4, 2)
    ]
    sf = simulate_knockout_round(sf_matchups, df, elo_ratings, predict_fn, match_date)

    final_matchup = [{"home": sf[0]["winner"], "away": sf[1]["winner"]}]
    final = simulate_knockout_round(final_matchup, df, elo_ratings, predict_fn, match_date)

    return {
        "groups": group_results,
        "knockout": {
            "r32": r32,
            "r16": r16,
            "qf": qf,
            "sf": sf,
            "final": final[0],
        },
        "champion": final[0]["winner"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bracket.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/simulation/bracket.py tests/test_bracket.py
git commit -m "feat: add knockout simulation and tournament orchestrator"
```

---

### Task 5: API Endpoint

**Files:**
- Modify: `src/api/main.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing test for the /simulate-bracket endpoint**

The existing `mock_model_and_data` fixture only has 3 teams. We need a fixture with all 48 WC teams. Add a new fixture and test to `tests/test_api.py`:

```python
from src.data.groups import GROUPS


@pytest.fixture
def mock_bracket_client():
    """Client with all 48 World Cup teams for bracket simulation."""
    all_teams = set()
    for teams in GROUPS.values():
        all_teams.update(teams)

    # Build minimal fake match history with all teams
    rows = []
    teams_list = sorted(all_teams)
    for i in range(len(teams_list)):
        for j in range(i + 1, min(i + 3, len(teams_list))):
            rows.append({
                "date": pd.to_datetime("2020-01-01") + pd.Timedelta(days=i + j),
                "home_team": teams_list[i],
                "away_team": teams_list[j],
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "neutral": False,
            })
    fake_df = pd.DataFrame(rows)

    fake_probs = pd.DataFrame({
        "away_win": [0.25],
        "draw": [0.30],
        "home_win": [0.45],
    })

    with patch("src.api.main.load_model_and_data") as mock_load:
        mock_load.return_value = {
            "predict_fn": lambda X: fake_probs,
            "df": fake_df,
            "model_type": "xgboost",
            "teams": all_teams,
            "elo_ratings": compute_elo_ratings(fake_df),
        }
        from src.api.main import app
        with TestClient(app) as client:
            yield client


def test_simulate_bracket_returns_full_bracket(mock_bracket_client):
    client = mock_bracket_client
    resp = client.post("/simulate-bracket")
    assert resp.status_code == 200
    data = resp.json()
    assert "groups" in data
    assert len(data["groups"]) == 12
    assert "knockout" in data
    assert "r32" in data["knockout"]
    assert len(data["knockout"]["r32"]) == 16
    assert "r16" in data["knockout"]
    assert len(data["knockout"]["r16"]) == 8
    assert "qf" in data["knockout"]
    assert len(data["knockout"]["qf"]) == 4
    assert "sf" in data["knockout"]
    assert len(data["knockout"]["sf"]) == 2
    assert "final" in data["knockout"]
    assert "champion" in data
    assert isinstance(data["champion"], str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_simulate_bracket_returns_full_bracket -v`
Expected: FAIL (404 — endpoint doesn't exist yet)

- [ ] **Step 3: Add the /simulate-bracket endpoint to main.py**

Add these imports to the top of `src/api/main.py`:

```python
from src.simulation.bracket import simulate_tournament
```

Add this endpoint after the existing `/predict` endpoint:

```python
@app.post("/simulate-bracket")
def simulate_bracket():
    return simulate_tournament(
        df=_state["df"],
        elo_ratings=_state["elo_ratings"],
        predict_fn=_state["predict_fn"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/main.py tests/test_api.py
git commit -m "feat: add /simulate-bracket API endpoint"
```

---

### Task 6: Frontend — Tab System

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Add tab bar HTML after the nav**

Replace the `<main>` opening tag and add a tab bar. The existing content becomes the "predictor" tab, and a new empty div becomes the "bracket" tab.

In `static/index.html`, replace:
```html
  <main class="max-w-4xl mx-auto px-6 py-8 space-y-7">
```

with:
```html
  <!-- Tab Bar -->
  <div class="max-w-4xl mx-auto px-6 pt-6">
    <div class="flex gap-1 bg-white/5 rounded-xl p-1 border border-white/10">
      <button id="tab-predictor" onclick="switchTab('predictor')" class="tab-btn flex-1 py-2.5 rounded-lg text-sm font-semibold transition-all text-white bg-white/10">Match Predictor</button>
      <button id="tab-bracket" onclick="switchTab('bracket')" class="tab-btn flex-1 py-2.5 rounded-lg text-sm font-semibold transition-all text-slate-400 hover:text-slate-300">Tournament Bracket</button>
    </div>
  </div>

  <main id="tab-content-predictor" class="max-w-4xl mx-auto px-6 py-8 space-y-7">
```

Then wrap the closing `</main>` and add the bracket tab container. Replace:
```html
  </main>
```

with:
```html
  </main>

  <!-- Bracket Tab (hidden by default) -->
  <main id="tab-content-bracket" class="hidden mx-auto px-6 py-8" style="max-width: 1400px;">
    <div class="bg-surface border border-border rounded-2xl p-7 text-center">
      <h2 class="text-white font-semibold text-lg mb-3">2026 World Cup — Full Tournament Prediction</h2>
      <p class="text-sm text-slate-500 mb-6">Simulate the entire tournament from group stage to the final</p>
      <button id="simulate-btn" onclick="simulateTournament()" class="bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-400 hover:to-emerald-500 text-white px-8 py-3 rounded-xl font-semibold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed">
        Simulate Tournament
      </button>
      <div id="bracket-error" class="hidden text-red-400 text-sm mt-4"></div>
    </div>

    <!-- Groups Section (populated by JS) -->
    <div id="groups-section" class="hidden mt-7">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-white font-semibold text-lg">Group Stage Results</h3>
        <button onclick="toggleGroups()" id="groups-toggle" class="text-sm text-slate-400 hover:text-slate-300">Hide Groups</button>
      </div>
      <div id="groups-grid" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"></div>
    </div>

    <!-- Bracket Section (populated by JS) -->
    <div id="bracket-section" class="hidden mt-7">
      <h3 class="text-white font-semibold text-lg mb-4">Knockout Bracket</h3>
      <div id="bracket-container" class="overflow-x-auto"></div>
    </div>

    <!-- Champion Banner -->
    <div id="champion-banner" class="hidden mt-7 bg-gradient-to-r from-amber-500/20 to-yellow-500/20 border border-amber-500/30 rounded-2xl p-8 text-center">
      <div class="text-sm text-amber-400 uppercase tracking-wide mb-2">Predicted Champion</div>
      <div id="champion-name" class="text-4xl font-bold text-amber-300"></div>
    </div>
  </main>
```

- [ ] **Step 2: Add tab switching JavaScript**

Add at the start of the existing `<script>` tag (before the FIXTURES const):

```javascript
    // Tab switching
    function switchTab(tab) {
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('text-white', 'bg-white/10');
        btn.classList.add('text-slate-400');
      });
      document.getElementById('tab-' + tab).classList.add('text-white', 'bg-white/10');
      document.getElementById('tab-' + tab).classList.remove('text-slate-400');

      document.getElementById('tab-content-predictor').classList.toggle('hidden', tab !== 'predictor');
      document.getElementById('tab-content-bracket').classList.toggle('hidden', tab !== 'bracket');
    }

    function toggleGroups() {
      const grid = document.getElementById('groups-grid');
      const btn = document.getElementById('groups-toggle');
      grid.classList.toggle('hidden');
      btn.textContent = grid.classList.contains('hidden') ? 'Show Groups' : 'Hide Groups';
    }
```

- [ ] **Step 3: Verify manually**

Run: `cd /c/worldCup && python -m uvicorn src.api.main:app --reload`
Open `http://localhost:8000` in browser. Verify two tabs appear and clicking switches between them. The bracket tab should show the "Simulate Tournament" button.

- [ ] **Step 4: Commit**

```bash
git add static/index.html
git commit -m "feat: add tab system with bracket tab skeleton"
```

---

### Task 7: Frontend — Group Stage Display

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Add group rendering JavaScript**

Add after the `toggleGroups` function in the `<script>` tag:

```javascript
    let bracketData = null;

    async function simulateTournament() {
      const btn = document.getElementById('simulate-btn');
      const errorEl = document.getElementById('bracket-error');
      errorEl.classList.add('hidden');
      btn.disabled = true;
      btn.textContent = 'Simulating...';

      try {
        const res = await fetch('/simulate-bracket', { method: 'POST' });
        if (!res.ok) {
          const data = await res.json();
          throw new Error(data.detail || 'Simulation failed');
        }
        bracketData = await res.json();
        renderGroups(bracketData.groups);
        renderBracket(bracketData.knockout);
        renderChampion(bracketData.champion);
      } catch (e) {
        errorEl.textContent = e.message;
        errorEl.classList.remove('hidden');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Simulate Tournament';
      }
    }

    function renderGroups(groups) {
      const grid = document.getElementById('groups-grid');
      grid.innerHTML = '';

      Object.entries(groups).sort().forEach(([letter, data]) => {
        const table = document.createElement('div');
        table.className = 'bg-white/[0.03] border border-white/[0.06] rounded-xl p-4';
        let html = `<div class="text-white font-semibold text-sm mb-3">Group ${letter}</div>`;
        html += `<table class="w-full text-xs">
          <thead><tr class="text-slate-500">
            <th class="text-left pb-2">Team</th>
            <th class="text-center pb-2">W</th>
            <th class="text-center pb-2">D</th>
            <th class="text-center pb-2">L</th>
            <th class="text-center pb-2">GD</th>
            <th class="text-center pb-2">Pts</th>
          </tr></thead><tbody>`;

        data.standings.forEach(row => {
          const advClass = row.advanced
            ? (row.pos <= 2 ? 'text-emerald-400' : 'text-sky-400')
            : 'text-slate-400';
          const bg = row.advanced ? 'bg-white/[0.03]' : '';
          html += `<tr class="${bg}">
            <td class="${advClass} py-1 text-left font-medium">${row.team}</td>
            <td class="text-center text-slate-400">${row.w}</td>
            <td class="text-center text-slate-400">${row.d}</td>
            <td class="text-center text-slate-400">${row.l}</td>
            <td class="text-center text-slate-400">${row.gd > 0 ? '+' : ''}${row.gd}</td>
            <td class="text-center text-white font-semibold">${row.pts}</td>
          </tr>`;
        });

        html += '</tbody></table>';
        table.innerHTML = html;
        grid.appendChild(table);
      });

      document.getElementById('groups-section').classList.remove('hidden');
    }

    function renderChampion(champion) {
      document.getElementById('champion-name').textContent = champion;
      document.getElementById('champion-banner').classList.remove('hidden');
    }
```

- [ ] **Step 2: Verify manually**

Run the server and click "Simulate Tournament". Group tables should appear in a grid with advancing teams highlighted in green (top 2) and blue (third-place qualifiers).

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat: add group stage rendering and simulation trigger"
```

---

### Task 8: Frontend — Knockout Bracket Tree

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Add bracket CSS**

Add to the `<style>` block in `<head>`:

```css
    .bracket-round { display: flex; flex-direction: column; justify-content: center; gap: 8px; min-width: 180px; }
    .bracket-match { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 6px 10px; font-size: 12px; }
    .bracket-match .winner { color: #4ade80; font-weight: 600; }
    .bracket-match .loser { color: #64748b; }
    .bracket-match .prob { color: #94a3b8; font-size: 10px; margin-left: 4px; }
    .bracket-wrapper { display: flex; align-items: center; gap: 16px; padding: 16px 0; }
    .round-label { text-align: center; font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; font-weight: 600; }
```

- [ ] **Step 2: Add bracket rendering JavaScript**

Add the `renderBracket` function after `renderChampion` in the `<script>` tag:

```javascript
    function renderBracket(knockout) {
      const container = document.getElementById('bracket-container');
      container.innerHTML = '';

      const rounds = [
        { label: 'Round of 32', matches: knockout.r32, left: knockout.r32.slice(0, 8), right: knockout.r32.slice(8, 16) },
        { label: 'Round of 16', matches: knockout.r16, left: knockout.r16.slice(0, 4), right: knockout.r16.slice(4, 8) },
        { label: 'Quarterfinals', matches: knockout.qf, left: knockout.qf.slice(0, 2), right: knockout.qf.slice(2, 4) },
        { label: 'Semifinals', matches: knockout.sf, left: [knockout.sf[0]], right: [knockout.sf[1]] },
      ];

      const wrapper = document.createElement('div');
      wrapper.className = 'bracket-wrapper';

      // Left bracket (flows left → right)
      rounds.forEach((round, ri) => {
        const col = createRoundColumn(round.label, round.left, ri);
        wrapper.appendChild(col);
      });

      // Final in the center
      const finalCol = document.createElement('div');
      finalCol.className = 'bracket-round';
      finalCol.innerHTML = `<div class="round-label">Final</div>`;
      finalCol.appendChild(createMatchCard(knockout.final));
      wrapper.appendChild(finalCol);

      // Right bracket (flows right → left, so we reverse round order)
      for (let ri = rounds.length - 1; ri >= 0; ri--) {
        const round = rounds[ri];
        const col = createRoundColumn(round.label, round.right, ri);
        wrapper.appendChild(col);
      }

      container.appendChild(wrapper);
      document.getElementById('bracket-section').classList.remove('hidden');
    }

    function createRoundColumn(label, matches, roundIndex) {
      const col = document.createElement('div');
      col.className = 'bracket-round';
      // Increase vertical spacing for later rounds
      col.style.gap = (8 * Math.pow(2, roundIndex)) + 'px';
      col.innerHTML = `<div class="round-label">${label}</div>`;
      matches.forEach(m => col.appendChild(createMatchCard(m)));
      return col;
    }

    function createMatchCard(match) {
      const card = document.createElement('div');
      card.className = 'bracket-match';
      const isHomeWinner = match.winner === match.home;
      const homeClass = isHomeWinner ? 'winner' : 'loser';
      const awayClass = isHomeWinner ? 'loser' : 'winner';
      const homeProb = Math.round(match.home_win * 100);
      const awayProb = Math.round(match.away_win * 100);
      card.innerHTML = `
        <div class="flex justify-between items-center mb-1">
          <span class="${homeClass}">${match.home}</span>
          <span class="prob">${homeProb}%</span>
        </div>
        <div class="flex justify-between items-center">
          <span class="${awayClass}">${match.away}</span>
          <span class="prob">${awayProb}%</span>
        </div>
      `;
      return card;
    }
```

- [ ] **Step 3: Verify manually**

Run server, go to Bracket tab, click Simulate. Full bracket tree should display:
- Left side: 8 R32 → 4 R16 → 2 QF → 1 SF
- Center: Final
- Right side: 1 SF → 2 QF → 4 R16 → 8 R32
- Winners in green, losers dimmed
- Champion banner in gold/amber at bottom

- [ ] **Step 4: Commit**

```bash
git add static/index.html
git commit -m "feat: add knockout bracket tree rendering"
```

---

### Task 9: Final Integration Testing

**Files:**
- Read: `tests/test_bracket.py`, `tests/test_api.py`

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 2: Run the full app manually and verify end-to-end**

Run: `cd /c/worldCup && python -m uvicorn src.api.main:app --reload`

Verify:
1. Both tabs work (Match Predictor and Tournament Bracket)
2. Match Predictor still works as before
3. Clicking "Simulate Tournament" shows groups + bracket + champion
4. Group tables show standings with advancing teams highlighted
5. Bracket tree renders all rounds from R32 to Final
6. Winners are highlighted in green, champion in gold

- [ ] **Step 3: Commit any final fixes if needed**

```bash
git add -A
git commit -m "fix: integration fixes for bracket prediction"
```
